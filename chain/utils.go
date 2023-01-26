package chain

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	shell "github.com/ipfs/go-ipfs-api"
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

func UploadIPFS(data []byte) string {
	sh := shell.NewShell("localhost:5001")
	content := bytes.NewBuffer(data)
	hash, err := sh.Add(content)
	if err != nil {
		panic(errors.New(fmt.Sprintf("Error in uploading to IPFS, %s", err)))
	}
	fmt.Println("File uploaded to IPFS")
	return hash
}

func DownloadIPFS(address string) []byte {
	sh := shell.NewShell("localhost:5001")
	read, err := sh.Cat(address)
	if err != nil {
		panic(errors.New(fmt.Sprintf("Error in downloading from IPFS, %s", err)))
	}
	content, _ := ioutil.ReadAll(read)
	return content
}

func SaveBlock(node_id int, block Block) {
	dirname := fmt.Sprintf("results/node_%d", node_id)
	filename := fmt.Sprintf("%s/chain_block_%d.json", dirname, block.Index)
	var f *os.File
	if _, err := os.Stat(dirname); os.IsNotExist(err) {
		_ = os.MkdirAll(dirname, os.ModePerm)
	}
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		f, err = os.Create(filename)
		if err != nil {
			panic(err)
		}
	} else {
		f, _ = os.OpenFile(filename, os.O_RDWR|os.O_APPEND, 0644)
	}
	js_lastblock, _ := json.Marshal(block)
	w := bufio.NewWriter(f)
	w.Write(js_lastblock) //write the last block into the file to release the memory
	w.WriteString("\n")
	w.Flush()
	f.Close()
}
