package LDA_go

import (
	"testing"
)

func TestLDATraining(t *testing.T) {
	documents := []string{
		"Go is a statically typed, compiled programming language.",
		"Python is an interpreted, high-level programming language.",
		"Java is a class-based, object-oriented programming language.",
		"JavaScript is a programming language that conforms to the ECMAScript specification.",
		"C is a general-purpose, procedural computer programming language.",
		"Ruby is an interpreted, high-level, general-purpose programming language.",
	}
	numTopics := 2
	numIterations := 100
	alpha := 25.0
	beta := 0.01
	model := NewLDA(numTopics, alpha, beta)
	err := model.Train(documents, numIterations)
	if err != nil {
		t.Errorf("Error training LDA model: %v", err)
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

func TestComputeTopicDistribution(t *testing.T) {
	// Prepare a minimal LDA model for testing
	alpha := 25.0
	beta := 0.01
	model := NewLDA(2, alpha, beta)
	model.vocabularySize = 3
	model.numTopics = 2
	model.beta = 0.1
	model.alpha = 0.1
	model.topicWordCounts = [][]int{
		{1, 2, 3},
		{4, 5, 6},
	}
	model.topicCounts = []int{6, 15}
	model.docTopicCounts = [][]int{
		{2, 3},
	}
	model.docLengths = []int{5}

	distribution := model.computeTopicDistribution(0, 1)
	if len(distribution) != 2 {
		t.Errorf("Expected distribution of length 2, got %d", len(distribution))
	}
	if distribution[0] <= 0 || distribution[1] <= 0 {
		t.Errorf("Expected positive probabilities, got %v", distribution)
	}
}

func TestSample(t *testing.T) {
	distribution := []float64{0.1, 0.2, 0.3, 0.4}
	counts := make([]int, len(distribution))
	trials := 10000
	for i := 0; i < trials; i++ {
		idx := sample(distribution)
		if idx < 0 || idx >= len(distribution) {
			t.Errorf("Sampled index out of bounds: %d", idx)
		}
		counts[idx]++
	}
	for i, count := range counts {
		expected := distribution[i] / 1.0 * float64(trials)
		if float64(count) < expected*0.9 || float64(count) > expected*1.1 {
			t.Errorf("Sample count for index %d out of expected range: got %d, expected around %.0f", i, count, expected)
		}
	}
}
