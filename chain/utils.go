package chain

import (
	"encoding/binary"
	"math/rand"
	"time"
)

func Int64ToBytes(i int64) []byte {
	var buf = make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(i))
	return buf
}

func Random(low int, high int) int {
	rand.Seed(time.Now().Unix())
	return rand.Intn(high-low) + low
}

func RandomArray(length int, low int, high int) []int {
	rand.Seed(time.Now().Unix())
	randArray := rand.Perm(high - low)
	for i := 0; i < len(randArray); i++ {
		randArray[i] = randArray[i] + low
	}
	return randArray[:length]
}
