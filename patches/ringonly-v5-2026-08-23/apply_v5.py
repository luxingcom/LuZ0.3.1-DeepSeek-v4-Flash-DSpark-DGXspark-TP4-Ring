#!/usr/bin/env python3
"""Apply v5 ring-order forcing + v1b physical-adjacency filter to NCCL 2.30.7 hardened tree.
Run from repo root (clone of nccl-official-2307-hardened-20260816, HEAD 0dd44cd).
ADR-016. Fails loudly if anchors are not found exactly once.
"""
import sys

# ---------- v1b: src/transport.cc ----------
p = 'src/transport.cc'
s = open(p).read()
old = """  int ringPrev = channel->ring.prev;
  int ringNext = channel->ring.next;
"""
new = """  int ringPrev = channel->ring.prev;
  int ringNext = channel->ring.next;
  // RING-ONLY PATCH v1b (ADR-016): on the 4-node switchless ring (nNodes==4,
  // nRanks==4, 1 rank/node) filter inter-node peers by HARDCODED physical
  // adjacency instead of this channel's ring order. Rank order 0-1-2-3-0
  // equals the physical ring 01-02-04-03 (same assumption as the v4 per-peer
  // dev map, validated in production). Defense-in-depth with v5 (connect.cc):
  // a channel carrying a non-physical ring order would pass the original v1
  // filter on non-adjacent "ring neighbors" that have no physical path.
  if (comm->nRanks == 4 && comm->nNodes == 4) {
    ringPrev = (comm->rank + comm->nRanks - 1) % comm->nRanks;
    ringNext = (comm->rank + 1) % comm->nRanks;
  }
"""
assert s.count(old) == 1, 'v1b anchor not found (count=%d)' % s.count(old)
open(p, 'w').write(s.replace(old, new))
print('v1b applied ->', p)

# ---------- v5: src/graph/connect.cc ----------
p = 'src/graph/connect.cc'
s = open(p).read()
anchor = """    for (int r = 0; r < nranks; r++) {
      ringPrev[c * nranks + r] = allTopoRanks[r]->ringPrev[c];
      ringNext[c * nranks + r] = allTopoRanks[r]->ringNext[c];
    }
  }
"""
ins = anchor + """
  // RING-ONLY PATCH v5 (ADR-016): force every channel onto the physical ring.
  // Cluster: switchless 4-node ring 01-02-04-03 (RoCE, 4x200G, 1 rank/node);
  // in rank space the physical ring is 0->1->2->3->0 (same assumption as the
  // v4 per-peer dev map running in production). NCCL 2.30 rail alternation
  // (crossNicRing==2 channel swap above) and multi-ring graph search can
  // produce non-physical ring orders once nChannels > 4; the v4 netdev
  // hardcode then hits a non-adjacent pair ({0,0} sentinel -> auto dev
  // selection -> QP without physical path -> constant ~17-20ms/AR stall at
  // 8/16 channels). Overwrite ringPrev/ringNext (all channels, all ranks)
  // and ringRecv/ringSend (node level) to node order, which equals the
  // physical ring here. Downstream consumers (connectRings, channel->ring,
  // the duplicate memcpy, copyChannels, ncclBuildRings) all derive from
  // these arrays, so plan ring == connect ring. Gated to nNodes==4 &&
  // nRanks==4 (other topologies keep stock behavior); NCCL_RINGONLY_V5=0
  // disables for A/B testing.
  if (comm->nNodes == 4 && comm->nRanks == 4) {
    const char* v5env = ncclGetEnv("NCCL_RINGONLY_V5");
    if (v5env == NULL || v5env[0] != '0') {
      for (int c = 0; c < nChannels; c++) {
        for (int n = 0; n < nNodes; n++) {
          int r = firstRanks[n];
          ringRecv[c * nNodes + n] = r;
          ringSend[c * nNodes + n] = r;
          ringPrev[c * nranks + r] = firstRanks[(n + nNodes - 1) % nNodes];
          ringNext[c * nranks + r] = firstRanks[(n + 1) % nNodes];
        }
        INFO(NCCL_INIT, "RING-ONLY v5 rank %d chan %d forced ring prev %d next %d",
             comm->rank, c, ringPrev[c * nranks + comm->rank], ringNext[c * nranks + comm->rank]);
      }
    }
  }
"""
assert s.count(anchor) == 1, 'v5 anchor not found (count=%d)' % s.count(anchor)
open(p, 'w').write(s.replace(anchor, ins))
print('v5 applied ->', p)
print('OK')
