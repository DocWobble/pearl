// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/stretchr/testify/assert"
)

func TestIsWalletSend(t *testing.T) {
	assert.False(t, isWalletSend(&btcjson.GetTransactionResult{
		Details: []btcjson.GetTransactionDetailsResult{{Category: "receive"}},
	}))
	assert.True(t, isWalletSend(&btcjson.GetTransactionResult{
		Details: []btcjson.GetTransactionDetailsResult{
			{Category: "send"},
			{Category: "receive"},
		},
	}))
	assert.False(t, isWalletSend(&btcjson.GetTransactionResult{}))
}
