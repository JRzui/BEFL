package network

import (
	"log"
	"sync"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BCFedMI/chain"
	"github.com/JRzui/BCFedMI/consensus"
	"github.com/coniks-sys/coniks-go/crypto/vrf"
	"github.com/pkg/errors"
)

type BlockchainNetwork struct {
	Transactions   []chain.LocalTransaction //pending transactions
	CandidateBlock []chain.Block            //pending block
	VerifiedBlock  chain.Block              //the verfied block
	Members        []chain.Member           //committee members
	Nodes          chain.PeerSet            //the registered nodes in the network
	Task           TaskInfo                 //the task published in the network

	VoteLock      sync.Mutex //control the voting process
	CommitteeLock sync.Mutex //control the committee process
	BlockLock     sync.Mutex //control the candidate block receiving process
	TxLock        sync.Mutex //control the tx receiving process

	CommitteeSetup bool
	BlockReceived  bool
	NewBlock       bool
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
	GlobalParam   [][]float64
	Momentum      [][]float64
	ModelSize     [][]int
	CompModelSize []int //the compressed model size
	PartNum       int   //the number of participants required in each training round
	Rank          int
	Beta          float64
	Slr           float64
	UnlabeledData *python3.PyObject
}

func NetworkInit() *BlockchainNetwork {
	network := &BlockchainNetwork{
		Transactions:   make([]chain.LocalTransaction, 0),
		CandidateBlock: make([]chain.Block, 0),
		VerifiedBlock:  chain.Genesis(),
		Members:        make([]chain.Member, 0),
		Nodes:          chain.NewPeerSet(),
		VoteLock:       sync.Mutex{},
		BlockLock:      sync.Mutex{},
		TxLock:         sync.Mutex{},
		CommitteeSetup: false,
		BlockReceived:  false,
		NewBlock:       false,
		NewTask:        false,
	}

	return network
}

func (bcnet *BlockchainNetwork) Register(nodeInfo RegisterInfo, reply *bool) error {
	*reply = bcnet.Nodes.Add(nodeInfo.ID, chain.Peer{nodeInfo.Address, nodeInfo.Pk})
	log.Printf("New node joins, total %d nodes in the network", len(bcnet.Nodes.Keys()))
	return nil
}

func (bcnet *BlockchainNetwork) SendBlock(block chain.Block, sent *bool) error {
	bcnet.BlockLock.Lock()
	bcnet.CandidateBlock = append(bcnet.CandidateBlock, block)
	bcnet.BlockReceived = true
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
	_, found := bcnet.Nodes.Set[member.ID]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	bcnet.CommitteeLock.Lock()
	if len(bcnet.Members) < chain.CommitteeSize {
		//verify the role information
		if consensus.ValidRole(bcnet.VerifiedBlock, member.Pk, member.VrfValue, member.VrfProof) {
			bcnet.Members = append(bcnet.Members, member)
		}
	}
	bcnet.CommitteeLock.Unlock()

	if len(bcnet.Members) == chain.CommitteeSize {
		bcnet.CommitteeSetup = true
	}
	*committeeSetup = bcnet.CommitteeSetup
	return nil
}

func (bcnet *BlockchainNetwork) CommitteeUpdate(id int, members *[]chain.Member) error {
	_, found := bcnet.Nodes.Set[id]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	*members = bcnet.Members
	return nil
}

func (bcnet *BlockchainNetwork) GetBlock(id int, block *chain.Block) error {
	_, found := bcnet.Nodes.Set[id]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	*block = bcnet.VerifiedBlock
	return nil
}

func (bcnet *BlockchainNetwork) GetTxs(id int, txs *[]chain.LocalTransaction) error {
	_, found := bcnet.Nodes.Set[id]
	if found == false {
		return errors.New("You have not registered for the network.")
	}

	*txs = bcnet.Transactions
	return nil
}
