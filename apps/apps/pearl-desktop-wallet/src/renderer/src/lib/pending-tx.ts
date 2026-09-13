import type { Transaction } from '../../../types/transaction';

// The daemon reports this when it announced a transaction but no peer asked
// for it. On a first send nothing left this machine; on a rebroadcast it is
// ambiguous, because peers that already hold the transaction stay silent.
export function isNotRelayedError(message: string): boolean {
  return message.includes('not relayed') || message.includes('no connected peers');
}

export const REBROADCAST_NOT_RELAYED_MESSAGE =
  'No peer requested it: either every connected peer already has it, or none will take it.';

export const REMOVE_WARNING =
  'Its inputs become spendable again, and any pending transaction spending from it is removed too. ' +
  'The network is not consulted: if a peer already holds this transaction it may still confirm, ' +
  'and spending the freed inputs again is a double-spend attempt.';

function formatAgo(timestampMs: number, nowMs: number): string {
  const minutes = Math.floor((nowMs - timestampMs) / 60_000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

// Status line for an unconfirmed transaction. Relay fields are absent on a
// full-node daemon, where there is nothing to say beyond "pending".
export function pendingStatusLabel(tx: Transaction, nowMs = Date.now()): string {
  if (tx.relayed === undefined) return 'Pending';
  if (!tx.relayed) return 'Pending, not announced since start';
  if (tx.lastRelayTime) return `Pending, relayed ${formatAgo(tx.lastRelayTime, nowMs)}`;
  return 'Pending, relayed';
}
