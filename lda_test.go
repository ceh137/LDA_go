package LDA_go

import (
	"fmt"
	"io/ioutil"
	"math"
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
	if len(model.Lambda) != numTopics {
		t.Errorf("Model lambda should be empty after reset")
	}
}

func TestTokenize(t *testing.T) {
	text := "Go is an open-source programming language."
	tokens := Tokenize(text)
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

// New test function added as per the user's request
func TestModelWorkflow(t *testing.T) {
	// Step 1: Create model
	numTopics := 5
	model := NewLDA(numTopics)

	// Step 2: Train it from file
	// Create temporary training data file
	trainingData1 := `Go is a statically typed, compiled programming language designed at Google.
Python is an interpreted, high-level, general-purpose programming language.
Java is a class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible.`
	tempFile1, err := ioutil.TempFile("", "training_data1_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile1.Name())
	if _, err := tempFile1.WriteString(trainingData1); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile1.Close()

	// Channels for iteration and error handling
	iterCh := make(chan int)
	errCh := make(chan error)

	// Start goroutine to monitor progress
	go func() {
		for {
			select {
			case iter, ok := <-iterCh:
				if !ok {
					iterCh = nil
				} else {
					t.Logf("Training iteration %d completed", iter)
				}
			case err, ok := <-errCh:
				if !ok {
					errCh = nil
				} else {
					t.Fatalf("Error during training: %v", err)
				}
			}
			if iterCh == nil && errCh == nil {
				break
			}
		}
	}()

	model.TrainFromFile(tempFile1.Name(), 20, iterCh, errCh)

	// Step 3: Save it to file
	modelFileName := "test_model_workflow.json"
	err = model.SaveModel(modelFileName)
	if err != nil {
		t.Fatalf("Error saving model: %v", err)
	}
	defer os.Remove(modelFileName)

	// Step 4: Update from another file
	trainingData2 := `JavaScript is a programming language that conforms to the ECMAScript specification.
C is a general-purpose, procedural computer programming language.
Ruby is an interpreted, high-level, general-purpose programming language.`
	tempFile2, err := ioutil.TempFile("", "training_data2_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempFile2.Name())
	if _, err := tempFile2.WriteString(trainingData2); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	tempFile2.Close()

	// Channels for iteration and error handling during update
	updateIterCh := make(chan int)
	updateErrCh := make(chan error)

	// Start goroutine to monitor progress
	go func() {
		for {
			select {
			case iter, ok := <-updateIterCh:
				if !ok {
					updateIterCh = nil
				} else {
					t.Logf("Update iteration %d completed", iter)
				}
			case err, ok := <-updateErrCh:
				if !ok {
					updateErrCh = nil
				} else {
					t.Fatalf("Error during update: %v", err)
				}
			}
			if updateIterCh == nil && updateErrCh == nil {
				break
			}
		}
	}()

	model.UpdateFromFile(tempFile2.Name(), 20, updateIterCh, updateErrCh)

	// Step 5: Save model to the same file
	err = model.SaveModel(modelFileName)
	if err != nil {
		t.Fatalf("Error saving updated model: %v", err)
	}

	// Step 6: Load the model from the file and score it
	loadedModel := NewLDA(numTopics)
	err = loadedModel.LoadModel(modelFileName)
	if err != nil {
		t.Fatalf("Error loading model from file: %v", err)
	}

	// Use the loaded model to get topics
	topics := loadedModel.GetTopics(5)
	if len(topics) != numTopics {
		t.Errorf("Expected %d topics, got %d", numTopics, len(topics))
	}
	for i, topic := range topics {
		if len(topic) != 5 {
			t.Errorf("Expected 5 words for topic %d, got %d", i, len(topic))
		}
		t.Logf("Topic %d: %v", i, topic)
	}

	// Score a new document
	newDocument := "Go and Python are popular programming languages."
	words := Tokenize(newDocument)
	wordIDs := make([]int, 0)
	for _, word := range words {
		if id, exists := loadedModel.Word2id[word]; exists {
			wordIDs = append(wordIDs, id)
		}
	}
	if len(wordIDs) == 0 {
		t.Errorf("No words from the new document are in the model's vocabulary.")
	}

	// Perform variational inference on the new document
	gamma, err := loadedModel.VariationalInference(wordIDs)
	if err != nil {
		t.Fatalf("Error during variational inference: %v", err)
	}

	// Check if gamma values are valid
	for k, val := range gamma {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			t.Errorf("Invalid gamma value for topic %d: %v", k, val)
		}
	}

	t.Logf("Gamma values for the new document: %v", gamma)
}
