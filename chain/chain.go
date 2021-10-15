package chain

import "reflect"

type Blockchain struct {
	Chain        []Block         `json:"chain"`        // linear chain structure
	Transactions TransactionPool `json:"transactions"` // the transactions
	Nodes        PeerSet         `json:"nodes"`        // the participated nodes
	Committee    MemberSet       `json:"committee"`    // the committee members
}

func NewBlockchain() *Blockchain {
	newBlockchain := &Blockchain{
		Chain:        make([]Block, 0),
		Transactions: NewTxPool(),
		Nodes:        NewPeerSet(),
		Committee:    NewMemberSet(),
	}
	newBlockchain.Chain = append(newBlockchain.Chain, Genesis())
	return newBlockchain
}

/*
Get the latest block in the chain
Returns:	-{Block} the latest block of the chain
*/

func (bc *Blockchain) LastBlock() Block {
	return bc.Chain[len(bc.Chain)-1]
}

/*
Get the length of the chain
Returns:	-{int} length
*/
func (bc *Blockchain) GetLength() int {
	return len(bc.Chain)
}

//Add block to the chain
func (bc *Blockchain) AddBlock(block Block) {
	bc.Chain = append(bc.Chain, block)
}

//Receive transaction and put it into the local transaction pool
func (bc *Blockchain) AddTransaction(tx LocalTransaction) {
	bc.Transactions.Add(tx)
}

//Update the committee group, called when the proof of role is valid
func (bc *Blockchain) CommitteeUpdate(member Member) bool {
	return bc.Committee.Add(member.ID, member)
}

//Clear the committee group, called when a new block is generated and wait for the next round committee construction
func (bc *Blockchain) CommitteeClear() {
	bc.Committee = NewMemberSet()
}

func ValidChain(localChain []Block, peerChain []Block) bool {
	localLen := len(localChain)
	for i := 0; i < localLen; i++ {
		//Compare if every block in the local chain is the same with the peers
		if !reflect.DeepEqual(localChain[i], peerChain[i]) {
			return false
		}
	}
	for i := localLen; i < len(peerChain); i++ {
		//Verify if the new blocks in the peer chain is correct
		if !ValidVerifiedBlock(peerChain[i-1], peerChain[i]) {
			return false
		}
	}
	return true
}
