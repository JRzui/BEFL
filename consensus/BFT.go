package consensus

import (
	"bufio"
	"crypto/elliptic"
	"encoding/gob"
	"log"
	"net"
	"strings"

	"github.com/JRzui/BEFL/chain"
)

func Vote(candidate chain.Block, members []chain.Member) chain.Block {
	yesVote := 0
	signatures := make([]chain.Signature, 0)
	for _, member := range members {
		conn, err := net.Dial("tcp", member.Address)
		if err != nil {
			log.Printf("Dialing %s failed, committee member is not online.\n", member.Address)
			continue
		}

		rw := bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn))
		rw.WriteString("Vote\n")
		enc := gob.NewEncoder(rw)
		err = enc.Encode(candidate)
		rw.Flush()

		//wait for reply
		vote, err := rw.ReadString('\n')
		if err != nil {
			log.Printf("Cannot read response from committee member %d", member.ID)
			continue
		}
		vote = strings.Trim(vote, "\n ")
		if vote == "Y" {
			yesVote += 1
			var data chain.Signature
			gob.Register(elliptic.P256()) //panic: gob: type not registered for interface: elliptic.p256Curve
			dec := gob.NewDecoder(rw)
			err := dec.Decode(&data)
			if err != nil {
				log.Println("Error decoding the signature info.")
			} else {
				signatures = append(signatures, data)
			}
		}
	}
	if yesVote > chain.CommitteeSize*2/3 {
		candidate.Signatures = signatures
		candidate.Transactions[0].Deltas = nil //clear the redundant local model updates
		return candidate
	}
	return chain.Block{}
}
