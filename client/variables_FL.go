package client

var (
	Task       = "femnist"
	K          = 5  //local training step
	B          = 64 //batch size
	Beta       = 0.9
	Lr         = 0.1 //client learning rate
	Slr        = 1.0 //server learning rate
	Round      = 500
	M          = 20  //the number of participants in each round
	Cm         = 0.2 //the fraction of malicious clients existed per round
	Rank       = 2   //the rank of the compressed matrix
	ClientsNum = 50
)
