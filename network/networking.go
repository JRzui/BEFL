package network

import (
	"bufio"
	"io"
	"log"
	"net"
	"strings"
	"sync"

	"github.com/pkg/errors"
)

func Open(addr string) (*bufio.ReadWriter, error) {
	// Dial the remote process.
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return nil, errors.Wrap(err, "Dialing "+addr+" failed")
	}
	return bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn)), nil
}

type HandleFunc func(*bufio.ReadWriter, net.Addr)

// Endpoint provides an endpoint to other processess
// that they can send data to.
type Endpoint struct {
	listener net.Listener
	handler  map[string]HandleFunc

	// Maps are not threadsafe, so we need a mutex to control access.
	m sync.Mutex
}

// NewEndpoint creates a new endpoint. To keep things simple,
// the endpoint listens on a fixed port number.
func NewEndpoint() *Endpoint {
	// Create a new Endpoint with an empty list of handler funcs.
	return &Endpoint{
		handler: map[string]HandleFunc{},
		m:       sync.Mutex{},
	}
}

// AddHandleFunc adds a new function for handling incoming data.
func (e *Endpoint) AddHandleFunc(name string, f HandleFunc) {
	e.m.Lock()
	e.handler[name] = f
	e.m.Unlock()
}

// Listen starts listening on the endpoint port on all interfaces.
// At least one handler function must have been added
// through AddHandleFunc() before.
func (e *Endpoint) Listen(Port string) error {
	var err error
	e.listener, err = net.Listen("tcp", Port)
	if err != nil {
		return errors.Wrapf(err, "Unable to listen on port %s\n", Port)
	}
	for {
		conn, err := e.listener.Accept()
		if err != nil {
			log.Println("Failed accepting a connection request:", err)
			continue
		}
		go e.handleMessages(conn)
	}
}

func (e *Endpoint) handleMessages(conn net.Conn) {
	// Wrap the connection into a buffered reader for easier reading.
	rw := bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn))
	defer conn.Close()

	// Read from the connection until EOF. Expect a command name as the
	// next input. Call the handler that is registered for this command.
	for {
		cmd, err := rw.ReadString('\n')
		switch {
		case err == io.EOF:
			log.Println("Reached EOF - close this connection. ---")
			return
		case err != nil:
			log.Println("Error reading command. Got: '"+cmd+"'\n", err)
			return
		}
		// Trim the request string - ReadString does not strip any newlines.
		cmd = strings.Trim(cmd, "\n ")

		// Fetch the appropriate handler function from the 'handler' map and call it.
		e.m.Lock()
		handleCommand, ok := e.handler[cmd]
		e.m.Unlock()
		if !ok {
			log.Println("Command '" + cmd + "' is not registered.")
			//return
		}
		address := conn.RemoteAddr()
		handleCommand(rw, address)
	}
}
