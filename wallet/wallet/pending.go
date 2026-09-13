package wallet

import (
	"errors"
	"fmt"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/walletdb"
	"github.com/pearl-research-labs/pearl/wallet/wtxmgr"
)

// ErrTxConfirmed is returned when an operation that only makes sense for a
// pending transaction is asked of one that is already mined.
var ErrTxConfirmed = errors.New("transaction is already confirmed")

// RelayStatus is the evidence the chain backend holds about whether the
// network took a pending transaction.
type RelayStatus struct {
	// Tracked is false when the backend keeps no such evidence, which is
	// the case for a full node: its own mempool answers the question and
	// nothing here should be shown to the user.
	Tracked bool

	// Relayed is true if a peer requested the transaction after an
	// announcement made during this daemon session. False therefore means
	// "not announced since start", not "the network lacks it".
	Relayed bool

	// LastRelayed is when that peer request happened; zero if !Relayed.
	LastRelayed time.Time
}

// broadcastTracker returns the chain backend's relay evidence, if it keeps
// any.
func (w *Wallet) broadcastTracker() (chain.BroadcastTracker, bool) {
	tracker, ok := w.ChainClient().(chain.BroadcastTracker)
	return tracker, ok
}

// RelayStatus reports the backend's relay evidence for txHash.
func (w *Wallet) RelayStatus(txHash chainhash.Hash) RelayStatus {
	tracker, ok := w.broadcastTracker()
	if !ok {
		return RelayStatus{}
	}

	last, relayed := tracker.LastRelayed(txHash)
	return RelayStatus{Tracked: true, Relayed: relayed, LastRelayed: last}
}

// pendingTxDetails loads txHash and rejects anything that is not a pending
// transaction of this wallet.
func (w *Wallet) pendingTxDetails(ns walletdb.ReadBucket,
	txHash chainhash.Hash) (*wtxmgr.TxDetails, error) {

	details, err := w.TxStore.TxDetails(ns, &txHash)
	if err != nil {
		return nil, err
	}
	if details == nil {
		return nil, fmt.Errorf("%w: txid %v", ErrNoTx, txHash)
	}
	if details.Block.Height != -1 {
		return nil, fmt.Errorf("%w: txid %v", ErrTxConfirmed, txHash)
	}

	return details, nil
}

// RemoveTransaction forgets the pending transaction txHash together with
// every pending transaction that spends from it, so the inputs they consumed
// become spendable again. It returns the removed hashes, txHash first.
//
// The network is not consulted: a peer that already holds the transaction
// may still mine it, in which case spending the freed inputs again is a
// double-spend attempt. Callers must have warned the user before calling.
func (w *Wallet) RemoveTransaction(txHash chainhash.Hash) ([]chainhash.Hash,
	error) {

	var removed []chainhash.Hash
	err := walletdb.Update(w.db, func(dbTx walletdb.ReadWriteTx) error {
		txmgrNs := dbTx.ReadWriteBucket(wtxmgrNamespaceKey)

		details, err := w.pendingTxDetails(txmgrNs, txHash)
		if err != nil {
			return err
		}

		unmined, err := w.TxStore.UnminedTxs(txmgrNs)
		if err != nil {
			return err
		}
		removed = dependentTxHashes(txHash, unmined)

		return w.TxStore.RemoveUnminedTx(txmgrNs, &details.TxRecord)
	})
	if err != nil {
		return nil, err
	}

	// Relay evidence must not outlive the record it describes: a later
	// transaction with the same hash would otherwise read as relayed.
	if tracker, ok := w.broadcastTracker(); ok {
		for _, hash := range removed {
			tracker.ForgetTransaction(hash)
		}
	}

	return removed, nil
}

// RebroadcastTransaction announces the pending transaction txHash to the
// network again, preceded by any of its ancestors that are still pending, in
// dependency order. Without a wallet-side retry loop a child whose parent no
// peer holds would otherwise stay an orphan forever.
//
// It stops at the first failure and returns the hashes announced so far along
// with the error. A chain.ErrTxNotRelayed keeps the record; a rejection
// removes it, as any resend does.
func (w *Wallet) RebroadcastTransaction(txHash chainhash.Hash) (
	[]chainhash.Hash, error) {

	var toAnnounce []*wire.MsgTx
	err := walletdb.View(w.db, func(dbTx walletdb.ReadTx) error {
		txmgrNs := dbTx.ReadBucket(wtxmgrNamespaceKey)

		details, err := w.pendingTxDetails(txmgrNs, txHash)
		if err != nil {
			return err
		}

		unmined, err := w.TxStore.UnminedTxs(txmgrNs)
		if err != nil {
			return err
		}
		toAnnounce = unminedAncestry(&details.MsgTx, unmined)

		return nil
	})
	if err != nil {
		return nil, err
	}

	announced := make([]chainhash.Hash, 0, len(toAnnounce))
	for _, tx := range toAnnounce {
		hash, err := w.publishTransaction(tx, republish)
		if err != nil {
			return announced, err
		}
		announced = append(announced, *hash)
	}

	return announced, nil
}

// dependentTxHashes returns root followed by every transaction in unmined
// that spends, directly or through other unmined transactions, an output of
// root.
func dependentTxHashes(root chainhash.Hash,
	unmined []*wire.MsgTx) []chainhash.Hash {

	hashes := make([]chainhash.Hash, len(unmined))
	for i, tx := range unmined {
		hashes[i] = tx.TxHash()
	}

	dependents := []chainhash.Hash{root}
	inSet := map[chainhash.Hash]struct{}{root: {}}
	for grew := true; grew; {
		grew = false
		for i, tx := range unmined {
			if _, ok := inSet[hashes[i]]; ok {
				continue
			}
			for _, txIn := range tx.TxIn {
				if _, ok := inSet[txIn.PreviousOutPoint.Hash]; !ok {
					continue
				}
				inSet[hashes[i]] = struct{}{}
				dependents = append(dependents, hashes[i])
				grew = true
				break
			}
		}
	}

	return dependents
}

// unminedAncestry returns tx together with every transaction in unmined that
// it depends on, directly or through other unmined transactions, sorted so
// that each transaction follows the ones it spends from.
func unminedAncestry(tx *wire.MsgTx, unmined []*wire.MsgTx) []*wire.MsgTx {
	byHash := make(map[chainhash.Hash]*wire.MsgTx, len(unmined))
	for _, u := range unmined {
		byHash[u.TxHash()] = u
	}

	ancestry := map[chainhash.Hash]*wire.MsgTx{tx.TxHash(): tx}
	queue := []*wire.MsgTx{tx}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, txIn := range cur.TxIn {
			parentHash := txIn.PreviousOutPoint.Hash
			parent, pending := byHash[parentHash]
			if !pending {
				continue
			}
			if _, seen := ancestry[parentHash]; seen {
				continue
			}
			ancestry[parentHash] = parent
			queue = append(queue, parent)
		}
	}

	return wtxmgr.DependencySort(ancestry)
}
