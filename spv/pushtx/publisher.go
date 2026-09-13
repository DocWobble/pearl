package pushtx

import (
	"sync"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/wire"
)

const (
	// DefaultBroadcastTimeout is the default timeout used when broadcasting
	// transactions to network peers.
	DefaultBroadcastTimeout = 5 * time.Second
)

// Publisher announces transactions to the network once per request and
// remembers, for the lifetime of the process, when a peer last took each one.
// Re-announcing is left to the caller: an SPV node cannot see mempools, so a
// wallet-side retry loop would only hide whether the network ever held the
// transaction from the user who has to decide what to do about it.
type Publisher struct {
	broadcast func(*wire.MsgTx) error

	mu          sync.Mutex
	lastRelayed map[chainhash.Hash]time.Time
}

// NewPublisher returns a Publisher that announces through broadcast, which
// must return a BroadcastError with code Mempool when peers already hold the
// transaction.
func NewPublisher(broadcast func(*wire.MsgTx) error) *Publisher {
	return &Publisher{
		broadcast:   broadcast,
		lastRelayed: make(map[chainhash.Hash]time.Time),
	}
}

// Publish announces tx once. A nil result or a Mempool error both mean the
// network holds the transaction and are recorded as relay evidence and
// reported as success. Every other error is returned unchanged and records
// nothing, so LastRelayed reflects only announcements a peer acted on.
func (p *Publisher) Publish(tx *wire.MsgTx) error {
	err := p.broadcast(tx)
	if err != nil && !IsBroadcastError(err, Mempool) {
		return err
	}

	p.mu.Lock()
	p.lastRelayed[tx.TxHash()] = time.Now()
	p.mu.Unlock()

	return nil
}

// LastRelayed reports when a peer last took txHash during this session. The
// bool is false if no announcement of txHash has succeeded since the
// Publisher was created or since Forget was last called for it.
func (p *Publisher) LastRelayed(txHash chainhash.Hash) (time.Time, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	t, ok := p.lastRelayed[txHash]
	return t, ok
}

// Forget drops the relay evidence for txHash. Callers use it once the
// transaction confirms or is removed from the wallet, so the evidence never
// outlives the record it describes.
func (p *Publisher) Forget(txHash chainhash.Hash) {
	p.mu.Lock()
	delete(p.lastRelayed, txHash)
	p.mu.Unlock()
}
