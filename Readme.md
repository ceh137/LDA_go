# Online (Incremental) Latent Dirichlet Allocation (LDA) Model 

This Go package provides a for-regular-human implementation of the Latent Dirichlet Allocation (LDA) algorithm for topic modeling in NLP.

### Features
- Simple API with minimalistic input with only text data and a few parameters.
- Understandable function names
- Parallelized using goroutines

### Installation:
Run Go get to download this package:
```bash
go get github.com/ceh137/LDA_go 
```
Import package into files you would like to use in:
```go
package main

import (
	lda "github.com/ceh137/LDA_go"
)
```

### Usage

#### Training Data Format
The training data should be provided in a **text file**, where each line represents a document. The structure is as follows:

- One document per line.
- Plain text format (UTF-8 encoded).
- No special formatting required.

**Example (training_data.txt):**
```text
Go is a statically typed, compiled programming language designed at Google.
Python is an interpreted, high-level, general-purpose programming language.
Java is a class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.
```

```go
package main

import (
	"fmt"
	lda "github.com/ceh137/LDA_go"
)

func main() {
	// Create a new LDA model
	numTopics := 5
	model := lda.NewLDA(numTopics)

	// Train the model from a training data file
	err := model.TrainFromFile("training_data.txt", 1000)
	if err != nil {
		fmt.Println("Error training LDA model:", err)
		return
	}

	// Get the topics
	topics := model.GetTopics(10) // Get top 10 words for each topic
	for i, topic := range topics {
		fmt.Printf("Topic %d: %v\n", i, topic)
	}

	// Save the model weights
	err = model.SaveModel("model_weights.json")
	if err != nil {
		fmt.Println("Error saving model:", err)
	}

	// Load the model weights
	err = model.LoadModel("model_weights.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
	}

	// Reset the model (delete weights and start from scratch)
	err = model.ResetModel()
	if err != nil {
		fmt.Println("Error resetting model:", err)
	}
}
```

### API Reference

`func NewLDA(numTopics int) *LDA`

Creates a new LDA model with the specified number of topics.

`func (model *LDA) TrainFromFile(filename string, numIterations int) error`

Trains the LDA model using the training data from the specified file for the given number of iterations.

`func (model *LDA) UpdateFromFile(filename string, numIterations int) error`

Updates the existing LDA model with new data from the specified file.

`func (model *LDA) GetTopics(numWords int) [][]string`

Returns the top numWords words for each topic.

`func (model *LDA) SaveModel(filename string) error`

Saves the model weights to a file in JSON format.

`func (model *LDA) LoadModel(filename string) error`

Loads the model weights from a file.

`func (model *LDA) ResetModel() error`

Deletes the existing model weights and resets the model to its initial state.

### Testing
Run the unit tests using:
```bash
go test 
```

### Features
- Online LDA Algorithm: Utilizes stochastic variational inference for incremental updates.
- Model Persistence: Weights are saved in a file.
- Parallelization: Goroutines and channels are used for efficient computation.

### Contributing

Contributions are welcome! Please open an issue or submit a pull request.