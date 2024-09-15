package LDA_go

import (
	"fmt"
	"os"
	"testing"
)

func TestLDATraining(t *testing.T) {
	documents := []string{
		"Go is a statically typed, compiled programming language designed at Google.",
		"Python is an interpreted, high-level, general-purpose programming language.",
		"Java is a class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.",
		"JavaScript is a programming language that conforms to the ECMAScript specification.",
		"C is a general-purpose, procedural computer programming language.",
		"Ruby is an interpreted, high-level, general-purpose programming language.",
	}
	numTopics := 2
	numIterations := 10

	model := NewLDA(numTopics)
	iterCh := make(chan int)
	errCh := make(chan error)
	go model.Train(documents, numIterations, iterCh, errCh)

	for {
		done := false
		select {
		case v, ok := <-iterCh:
			if !ok {
				done = true
				break
			}
			fmt.Println(v)
		case v, ok := <-errCh:
			if !ok {
				done = true
				break
			}
			t.Errorf("Expected no error, got %s", v)
			done = true
		}
		if done {
			break
		}
	}

	topics := model.GetTopics(5)
	if len(topics) != numTopics {
		t.Errorf("Expected %d topics, got %d", numTopics, len(topics))
	}
	for i, topic := range topics {
		if len(topic) != 5 {
			t.Errorf("Expected 5 words for topic %d, got %d", i, len(topic))
		}
	}

	// Test saving the model
	err := model.SaveModel("test_model.json")
	if err != nil {
		t.Errorf("Error saving model: %v", err)
	}
	defer os.Remove("test_model.json")

	// Test loading the model
	newModel := NewLDA(numTopics)
	err = newModel.LoadModel("test_model.json")
	if err != nil {
		t.Errorf("Error loading model: %v", err)
	}

	// Check if topics are the same
	newTopics := newModel.GetTopics(5)
	for i := range topics {
		for j := range topics[i] {
			if topics[i][j] != newTopics[i][j] {
				t.Errorf("Mismatch in topics after loading model")
				break
			}
		}
	}

	// Test resetting the model
	err = model.ResetModel()
	if err != nil {
		t.Errorf("Error resetting model: %v", err)
	}
	if len(model.lambda) != numTopics {
		t.Errorf("Model lambda should be empty after reset")
	}
}

func TestTokenize(t *testing.T) {
	text := "Go is an open-source programming language."
	tokens := tokenize(text)
	expected := []string{"go", "is", "an", "open", "source", "programming", "language"}
	if len(tokens) != len(expected) {
		t.Errorf("Expected %d tokens, got %d", len(expected), len(tokens))
	}
	for i, token := range tokens {
		if token != expected[i] {
			t.Errorf("Expected token '%s', got '%s'", expected[i], token)
		}
	}
}
