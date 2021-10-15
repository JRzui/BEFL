package chain

import "time"

var (
	CommitteeSize     = 5
	GossipPort        = "127.0.0.1:8000"
	NumTxs            = 5       // the maximum number of transactions that contained in the pool
	BaseNode          = ":8000" // the initial node in the blockchain network
	CommtteeThreshold = 0.5     // the vrf threshold
	VotingTimeout     = 30 * time.Second
	CommiteeTimeout   = 10 * time.Second
	BlockTimeout      = 15 * time.Second
)
