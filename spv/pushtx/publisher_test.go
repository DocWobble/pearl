package pushtx

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/stretchr/testify/require"
)

// txWithLockTime returns a distinct transaction per lockTime so tests can
// tell announcements apart by hash.
func txWithLockTime(lockTime uint32) *wire.MsgTx {
	tx := wire.NewMsgTx(wire.TxVersion)
	tx.LockTime = lockTime
	return tx
}

// TestPublisherRelayEvidence pins which broadcast outcomes count as the
// network having taken the transaction.
func TestPublisherRelayEvidence(t *testing.T) {
	notRelayed := &BroadcastError{Code: NotRelayed, Reason: "no takers"}
	rejected := &BroadcastError{Code: Invalid, Reason: "bad"}
	hardErr := errors.New("connection reset")

	tests := []struct {
		name        string
		broadcast   error
		wantErr     error
		wantRelayed bool
	}{
		{name: "accepted", broadcast: nil, wantRelayed: true},
		{
			name:        "already in mempool",
			broadcast:   &BroadcastError{Code: Mempool},
			wantRelayed: true,
		},
		{name: "not relayed", broadcast: notRelayed, wantErr: notRelayed},
		{name: "rejected", broadcast: rejected, wantErr: rejected},
		{name: "transport error", broadcast: hardErr, wantErr: hardErr},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := NewPublisher(func(*wire.MsgTx) error {
				return tc.broadcast
			})
			tx := txWithLockTime(1)

			before := time.Now()
			err := p.Publish(tx)
			require.Equal(t, tc.wantErr, err)

			last, ok := p.LastRelayed(tx.TxHash())
			require.Equal(t, tc.wantRelayed, ok)
			if tc.wantRelayed {
				require.False(t, last.Before(before))
				require.False(t, last.After(time.Now()))
			} else {
				require.True(t, last.IsZero())
			}
		})
	}
}

// TestPublisherForget checks that Forget clears exactly the named hash.
func TestPublisherForget(t *testing.T) {
	p := NewPublisher(func(*wire.MsgTx) error { return nil })
	kept, dropped := txWithLockTime(1), txWithLockTime(2)

	require.NoError(t, p.Publish(kept))
	require.NoError(t, p.Publish(dropped))

	p.Forget(dropped.TxHash())
	p.Forget(txWithLockTime(3).TxHash())

	_, ok := p.LastRelayed(dropped.TxHash())
	require.False(t, ok)
	_, ok = p.LastRelayed(kept.TxHash())
	require.True(t, ok)
}

// TestPublisherRepublishRefreshesEvidence checks a second successful
// announcement moves the timestamp forward rather than keeping the first.
func TestPublisherRepublishRefreshesEvidence(t *testing.T) {
	p := NewPublisher(func(*wire.MsgTx) error { return nil })
	tx := txWithLockTime(1)

	require.NoError(t, p.Publish(tx))
	first, ok := p.LastRelayed(tx.TxHash())
	require.True(t, ok)

	time.Sleep(time.Millisecond)
	require.NoError(t, p.Publish(tx))
	second, ok := p.LastRelayed(tx.TxHash())
	require.True(t, ok)
	require.True(t, second.After(first))
}

// TestPublisherConcurrent runs the whole API concurrently so the race
// detector can vet the locking.
func TestPublisherConcurrent(t *testing.T) {
	p := NewPublisher(func(*wire.MsgTx) error { return nil })

	var wg sync.WaitGroup
	for i := range 32 {
		tx := txWithLockTime(uint32(i))
		wg.Add(3)
		go func() {
			defer wg.Done()
			_ = p.Publish(tx)
		}()
		go func() {
			defer wg.Done()
			p.LastRelayed(tx.TxHash())
		}()
		go func() {
			defer wg.Done()
			p.Forget(tx.TxHash())
		}()
	}
	wg.Wait()
}
