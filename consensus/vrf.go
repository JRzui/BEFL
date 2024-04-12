package consensus

import (
	"fmt"
	"log"

	"github.com/JRzui/BEFL/chain"
	"github.com/coniks-sys/coniks-go/crypto/vrf"
)

type VRF struct {
	rolesSk vrf.PrivateKey // Used for the aggregator and miner VRF
	RolesPk vrf.PublicKey
}

func (myvrf *VRF) Init() {
	var err error
	myvrf.rolesSk, err = vrf.GenerateKey(nil) // create a publick/private key pair
	if err != nil {
		fmt.Println("Error, could not generate secret key for roles")
	}

	myvrf.RolesPk, _ = myvrf.rolesSk.Public()
}

func (myvrf *VRF) GetRolesPublicKey() vrf.PublicKey {
	return myvrf.RolesPk
}

func (myvrf *VRF) RolesCompute(input []byte) ([]byte, []byte) {
	/*
		Compute and return the vrf value and a proof
	*/
	return myvrf.rolesSk.Prove(input)
}

func Verify(input []byte, Pk vrf.PublicKey, inputVRF []byte, inputProof []byte) bool {
	return Pk.Verify(input, inputVRF, inputProof)
}

/*
Getting the role in consensus achievement:
whether the node itself become the member of the committee
Returns:
-{bool}	wheather being selected or not
-{[]byte}	the vrf random value
-{[]byte}	the proof of vrf value
*/
func (myvrf *VRF) GetRole(id int, Select int, stakeMap map[int]int, lastBlock chain.Block) (bool, []byte, []byte) {
	blockBytes := chain.BlockToByteWithSig(lastBlock)
	BlockHash := chain.ComputeHashForBlock(blockBytes)
	vrfValue, vrfProof := myvrf.RolesCompute(BlockHash) //calculate the vrf random value based on the latest block of the chain

	//the total stake in system
	stakeTotal := 0
	for id, _ := range stakeMap {
		stakeTotal += stakeMap[id]
	}
	ratio := HashRatio(vrfValue) / float64(stakeMap[id])
	if ratio < 3*float64(chain.CommitteeSize)/float64(stakeTotal) {
		return true, vrfValue, vrfProof
	}
	return false, vrfValue, vrfProof
}

func ValidRole(id int, Select int, stakeMap map[int]int, latestBlock chain.Block, pk vrf.PublicKey, vrfValue []byte, vrfProof []byte) bool {
	//verify whether the vrfValue is correct or not
	blockBytes := chain.BlockToByteWithSig(latestBlock)
	blockHash := chain.ComputeHashForBlock(blockBytes)
	check := Verify(blockHash, pk, vrfValue, vrfProof)
	if !check {
		log.Println("This node is not the member of the commitee, incorrect vrf proof")
		return false
	}
	//the total stake in system
	stakeTotal := 0
	for id, _ := range stakeMap {
		stakeTotal += stakeMap[id]
	}
	ratio := HashRatio(vrfValue) / float64(stakeMap[id])
	if ratio < 3*float64(chain.CommitteeSize)/float64(stakeTotal) {
		return true
	}
	return false
}
