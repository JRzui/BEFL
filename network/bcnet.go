package network

import (
	"fmt"
	"log"
	"sync"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BEFL/chain"
	"github.com/JRzui/BEFL/consensus"
	"github.com/coniks-sys/coniks-go/crypto/vrf"
	"github.com/pkg/errors"
)

type BlockchainNetwork struct {
	Transactions   []chain.LocalTransaction //pending transactions
	CandidateBlock chan chain.Block         //pending block
	BlockSender    map[string]int           //the sender of the candidate block
	VerifiedBlock  chain.Block              //the verfied block
	Members        chain.MemberSet          //committee members
	Nodes          chain.PeerSet            //the registered nodes in the network
	Task           TaskInfo                 //the task published in the network
	StakeMap       map[int]int              //the stake map of nodes
	Vrf            consensus.VRF            //stake based random role selection

	VoteLock      sync.Mutex //control the voting process
	CommitteeLock sync.Mutex //control the committee process
	BlockLock     sync.Mutex //control the candidate block receiving process
	TxLock        sync.Mutex //control the tx receiving process

	CommitteeWait  chan bool
	CommitteeSetup chan bool
	CandidateWait  chan bool
	BlockReceived  chan bool
	NewBlock       chan bool
	NewRound       chan bool
	NewTask        bool
}

type RegisterInfo struct {
	ID      int
	Address string
	Pk      vrf.PublicKey
}

//FL Task
type TaskInfo struct {
	TaskName      string
	Model         *python3.PyObject
	GlobalModel   string
	ModelSize     [][]int
	CompModelSize []int //the compressed model size
	PartNum       int   //the number of participants required in each training round
	Rank          int
	CurrentRound  int
	Beta          float64
	Slr           float64
	UnlabeledData *python3.PyObject
}

type Candidate struct {
	PendingBlock chain.Block
	Sender       int //the id number of the sender
}

func NetworkInit() *BlockchainNetwork {
	vrf := consensus.VRF{}
	vrf.Init()
	network := &BlockchainNetwork{
		Transactions:   make([]chain.LocalTransaction, 0),
		CandidateBlock: make(chan chain.Block),
		BlockSender:    make(map[string]int),
		VerifiedBlock:  chain.Genesis(),
		Members:        chain.NewMemberSet(),
		Nodes:          chain.NewPeerSet(),
		StakeMap:       make(map[int]int),
		Vrf:            vrf,
		CommitteeLock:  sync.Mutex{},
		VoteLock:       sync.Mutex{},
		BlockLock:      sync.Mutex{},
		TxLock:         sync.Mutex{},
		CommitteeWait:  make(chan bool),
		CommitteeSetup: make(chan bool),
		CandidateWait:  make(chan bool),
		BlockReceived:  make(chan bool),
		NewBlock:       make(chan bool),
		NewRound:       make(chan bool),
		NewTask:        false,
	}

	return network
}

func (bcnet *BlockchainNetwork) Register(nodeInfo RegisterInfo, reply *bool) error {
	*reply = bcnet.Nodes.Add(nodeInfo.ID, chain.Peer{nodeInfo.Address, nodeInfo.Pk})
	bcnet.StakeMap[nodeInfo.ID] = chain.DefaultStake
	log.Printf("New node joins, total %d nodes in the network", len(bcnet.Nodes.Keys()))
	return nil
}

func (bcnet *BlockchainNetwork) SendBlock(candi Candidate, sent *bool) error {
	bcnet.BlockLock.Lock()
	go func() { bcnet.CandidateBlock <- candi.PendingBlock }()
	bcnet.BlockSender[fmt.Sprintf("%x", candi.PendingBlock.Transactions[0].GlobalModelSig.Sig)] = candi.Sender
	go func() { bcnet.BlockReceived <- true }()
	*sent = true
	log.Println("Candidate block received")
	bcnet.BlockLock.Unlock()

	return nil
}

func (bcnet *BlockchainNetwork) SendTx(tx chain.LocalTransaction, sent *bool) error {
	bcnet.TxLock.Lock()
	bcnet.Transactions = append(bcnet.Transactions, tx)
	bcnet.TxLock.Unlock()

	*sent = true
	log.Println("Network: received new pending tx.")
	return nil
}

func (bcnet *BlockchainNetwork) SendRole(member chain.Member, committeeSetup *bool) error {
	bcnet.CommitteeLock.Lock()
	_, found := bcnet.Nodes.Set[member.ID]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	if len(bcnet.Members.Keys()) < chain.CommitteeSize {
		//verify the role information
		if consensus.ValidRole(member.ID, chain.CommitteeSize, bcnet.StakeMap, bcnet.VerifiedBlock, member.Pk, member.VrfValue, member.VrfProof) {
			bcnet.Members.Add(member.ID, member)
			fmt.Println("member ID ", member.ID)
		} else {
			fmt.Println("invalid role info")
		}
	}

	if len(bcnet.Members.Keys()) == chain.CommitteeSize {
		go func() { bcnet.CommitteeSetup <- true }()
		*committeeSetup = true
	} else {
		*committeeSetup = false
	}
	bcnet.CommitteeLock.Unlock()
	return nil
}

func (bcnet *BlockchainNetwork) CommitteeUpdate(id int, members *[]chain.Member) error {
	_, found := bcnet.Nodes.Set[id]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	_members := make([]chain.Member, 0)
	for _, member := range bcnet.Members.Members {
		_members = append(_members, member)
	}
	*members = _members
	return nil
}

func (bcnet *BlockchainNetwork) GetBlock(id int, block *chain.Block) error {
	_, found := bcnet.Nodes.Set[id]
	if !found {
		return errors.New("You have not registered for the network.")
	}

	*block = bcnet.VerifiedBlock
	return nil
}

func (bcnet *BlockchainNetwork) GetTxs(id int, txs *[]chain.LocalTransaction) error {
	_, found := bcnet.Nodes.Set[id]
	if !found {
		return errors.New("You have not registered for the network.")
	}

	*txs = bcnet.Transactions
	return nil
}

func RewardsDistribution(bcnet *BlockchainNetwork) {
	for _, member := range bcnet.Members.Members {
		bcnet.StakeMap[member.ID] += 1 //reward to committee members
		fmt.Println("node ", member.ID, "stake: ", bcnet.StakeMap[member.ID])
	}
	candCreater := bcnet.BlockSender[fmt.Sprintf("%x", bcnet.VerifiedBlock.Transactions[0].GlobalModelSig.Sig)]
	bcnet.StakeMap[candCreater] += 1 //reward to candidate block creater
	fmt.Println("node ", candCreater, "stake: ", bcnet.StakeMap[candCreater])
}
