# Latent Dirichlet Allocation (LDA) in Go

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
```go
package main

import (
    "fmt"
    lda "github.com/ceh137/LDA_go"
)

func main() {
    // Sample corpus
    documents := []string{
        "Go is a statically typed, compiled programming language.",
        "Python is an interpreted, high-level programming language.",
        "Java is a class-based, object-oriented programming language.",
        // Add more documents...
    }

    // Create a new LDA model
    numTopics := 2
    numIterations := 1000

	alpha := 25.0
	beta := 0.01

    model := lda.NewLDA(numTopics, alpha, beta)
    err := model.Train(documents, numIterations)
    if err != nil {
        fmt.Println("Error training LDA model:", err)
        return
    }

    // Get the topics
    topics := model.GetTopics(5) // Get top 5 words for each topic
    for i, topic := range topics {
        fmt.Printf("Topic %d: %v\n", i, topic)
    }
}
```

### API Reference
`func NewLDA(numTopics int, alpha, beta float64) *LDA`
Creates a new LDA model with the specified number of topics and params. 
There are default values for them in place `alpha=50/numTopic`, `beta=0.01`
To use them you have to provide `0.0` as input to the param.

`func (model *LDA) Train(documents []string, numIterations int) error`
Trains the LDA model on the provided documents.

`func (model *LDA) GetTopics(numWords int) [][]string`
Returns the top numWords words for each topic.