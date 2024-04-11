package chain

import "time"

var (
	CommitteeSize = 15      //5,10,15,20
	NumTxs        = 50       // the maximum number of transactions that contained in the pool
	BaseNode      = ":9000" // the initial node in the blockchain network
	DefaultStake  = 1
	NodeNum       = 50 //100, 200, 500, 1000
	SybilRatio    = 0.0 //the fraction of sybils in blockchain nodes
	//CommtteeThreshold = 0.5 * float64(DefaultStake) / float64(NodeNum) // the vrf threshold
	MaxVoteStep    = 5
	BlockTimeout   = 240 * time.Second
	VotingTimeout  = 200 * time.Second
	CommiteeSleep  = 20 * time.Second
	CandidateSleep = 20 * time.Second
)
