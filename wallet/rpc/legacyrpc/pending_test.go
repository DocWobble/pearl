package legacyrpc

import (
	"errors"
	"fmt"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/wallet/chain"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
	"github.com/stretchr/testify/require"
)

// TestPendingTxError pins the RPC codes clients branch on for the
// pending-transaction commands.
func TestPendingTxError(t *testing.T) {
	notRelayed := fmt.Errorf("%w: no peer requested", chain.ErrTxNotRelayed)

	tests := []struct {
		name     string
		err      error
		wantCode btcjson.RPCErrorCode
		wantMsg  string
	}{
		{
			name:     "unknown tx",
			err:      fmt.Errorf("%w: txid abc", wallet.ErrNoTx),
			wantCode: btcjson.ErrRPCNoTxInfo,
			wantMsg:  ErrNoTransactionInfo.Message,
		},
		{
			name:     "confirmed tx",
			err:      fmt.Errorf("%w: txid abc", wallet.ErrTxConfirmed),
			wantCode: btcjson.ErrRPCInvalidParameter,
			wantMsg:  "transaction is already confirmed: txid abc",
		},
		{
			name:     "not relayed keeps the reason",
			err:      notRelayed,
			wantCode: btcjson.ErrRPCInternal.Code,
			wantMsg:  notRelayed.Error(),
		},
		{
			name:     "other",
			err:      errors.New("db closed"),
			wantCode: btcjson.ErrRPCInternal.Code,
			wantMsg:  "db closed",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var rpcErr *btcjson.RPCError
			require.ErrorAs(t, pendingTxError(tc.err), &rpcErr)
			require.Equal(t, tc.wantCode, rpcErr.Code)
			require.Equal(t, tc.wantMsg, rpcErr.Message)
		})
	}
}

func TestPendingTxHash(t *testing.T) {
	_, err := pendingTxHash("not-hex")
	var rpcErr *btcjson.RPCError
	require.ErrorAs(t, err, &rpcErr)
	require.Equal(t, btcjson.ErrRPCDecodeHexString, rpcErr.Code)

	hash, err := pendingTxHash(
		"0000000000000000000000000000000000000000000000000000000000000001",
	)
	require.NoError(t, err)
	require.Equal(t, uint8(1), hash[0])
}
