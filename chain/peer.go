package chain

import "github.com/coniks-sys/coniks-go/crypto/vrf"

type Peer struct {
	Address string
	Pk      vrf.PublicKey
}

type PeerSet struct {
	Set   map[int]bool
	Peers map[int]Peer
}

type Member struct {
	ID       int
	Address  string
	Pk       vrf.PublicKey
	VrfValue []byte
	VrfProof []byte
}

type MemberSet struct {
	Set     map[int]bool
	Members map[int]Member
}

func NewPeerSet() PeerSet {
	return PeerSet{make(map[int]bool), map[int]Peer{}}
}

func NewMemberSet() MemberSet {
	return MemberSet{make(map[int]bool), map[int]Member{}}
}

/*
---Input:
string - the key(address) of node
---Ouput:
bool - true, the node has been successfully added
     - false, the node already existed
*/
func (set *PeerSet) Add(id int, peer Peer) bool {
	_, found := set.Set[id] // the found should return false
	set.Set[id] = true
	set.Peers[id] = peer
	return !found
}

func (set *PeerSet) Keys() []int {
	var keys []int
	for k, _ := range set.Set {
		keys = append(keys, k)
	}
	return keys
}

func (set *MemberSet) Add(id int, member Member) bool {
	_, found := set.Set[id] // the found should return false
	set.Set[id] = true
	set.Members[id] = member
	return !found
}

func (set *MemberSet) Keys() []int {
	var keys []int
	for k, _ := range set.Set {
		keys = append(keys, k)
	}
	return keys
}
