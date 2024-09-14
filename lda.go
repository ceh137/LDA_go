package LDA_go

import (
	"errors"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"
)

type LDA struct {
	numTopics      int
	alpha          float64
	beta           float64
	documents      [][]int
	vocabulary     []string
	word2id        map[string]int
	id2word        map[int]string
	vocabularySize int

	topicAssignments [][]int
	docTopicCounts   [][]int
	topicWordCounts  [][]int
	topicCounts      []int
	docLengths       []int
	totalDocs        int

	mutex sync.Mutex
}

// NewLDA creates a new LDA model
// Default values for alpha and beta are set to 50/numTopics and 0.01 respectively.
func NewLDA(numTopics int, alpha float64, beta float64) *LDA {
	if alpha == 0.0 {
		alpha = 50.0 / float64(numTopics)
	}
	if beta == 0.0 {
		beta = 0.01

	}
	return &LDA{
		numTopics: numTopics,
		alpha:     alpha,
		beta:      beta,
	}
}

// Train trains the LDA model on the provided documents for the number of iterations.
func (lda *LDA) Train(documents []string, numIterations int) error {
	if len(documents) == 0 {
		return errors.New("no documents provided for training")
	}

	lda.buildVocabulary(documents)
	lda.initializeStructures()
	lda.initializeAssignments()

	for iter := 0; iter < numIterations; iter++ {
		lda.gibbsSamplingIteration()
	}

	return nil
}

// GetTopics returns the top N words for each topic.
func (lda *LDA) GetTopics(numWords int) [][]string {
	topics := make([][]string, lda.numTopics)
	for k := 0; k < lda.numTopics; k++ {
		wordCounts := lda.topicWordCounts[k]
		wordProbabilities := make([]wordProbability, lda.vocabularySize)
		for w := 0; w < lda.vocabularySize; w++ {
			prob := (float64(wordCounts[w]) + lda.beta) / (float64(lda.topicCounts[k]) + float64(lda.vocabularySize)*lda.beta)
			wordProbabilities[w] = wordProbability{
				word:        lda.id2word[w],
				probability: prob,
			}
		}
		sort.Slice(wordProbabilities, func(i, j int) bool {
			return wordProbabilities[i].probability > wordProbabilities[j].probability
		})
		topWords := make([]string, numWords)
		for i := 0; i < numWords; i++ {
			topWords[i] = wordProbabilities[i].word
		}
		topics[k] = topWords
	}
	return topics
}

type wordProbability struct {
	word        string
	probability float64
}

// buildVocabulary builds the vocabulary and word mappings.
func (lda *LDA) buildVocabulary(documents []string) {
	wordSet := make(map[string]struct{})
	tokenizedDocs := make([][]string, len(documents))
	for i, doc := range documents {
		words := tokenize(doc)
		tokenizedDocs[i] = words
		for _, word := range words {
			wordSet[word] = struct{}{}
		}
	}

	lda.vocabulary = make([]string, 0, len(wordSet))
	lda.word2id = make(map[string]int)
	lda.id2word = make(map[int]string)
	id := 0
	for word := range wordSet {
		lda.vocabulary = append(lda.vocabulary, word)
		lda.word2id[word] = id
		lda.id2word[id] = word
		id++
	}
	lda.vocabularySize = len(lda.vocabulary)

	// Convert documents to word IDs
	lda.documents = make([][]int, len(tokenizedDocs))
	for i, words := range tokenizedDocs {
		wordIDs := make([]int, len(words))
		for j, word := range words {
			wordIDs[j] = lda.word2id[word]
		}
		lda.documents[i] = wordIDs
	}
}

// initializeStructures initializes the count matrices.
func (lda *LDA) initializeStructures() {
	lda.totalDocs = len(lda.documents)
	lda.topicAssignments = make([][]int, lda.totalDocs)
	lda.docTopicCounts = make([][]int, lda.totalDocs)
	lda.docLengths = make([]int, lda.totalDocs)
	for d := 0; d < lda.totalDocs; d++ {
		lda.docTopicCounts[d] = make([]int, lda.numTopics)
		lda.docLengths[d] = len(lda.documents[d])
	}
	lda.topicWordCounts = make([][]int, lda.numTopics)
	for k := 0; k < lda.numTopics; k++ {
		lda.topicWordCounts[k] = make([]int, lda.vocabularySize)
	}
	lda.topicCounts = make([]int, lda.numTopics)
}

// initializeAssignments initializes topic assignments randomly.
func (lda *LDA) initializeAssignments() {
	rand.Seed(time.Now().UnixNano())
	for d, doc := range lda.documents {
		lda.topicAssignments[d] = make([]int, len(doc))
		for n, wordID := range doc {
			topic := rand.Intn(lda.numTopics)
			lda.topicAssignments[d][n] = topic
			lda.docTopicCounts[d][topic]++
			lda.topicWordCounts[topic][wordID]++
			lda.topicCounts[topic]++
		}
	}
}

// gibbsSamplingIteration performs one iteration of Gibbs sampling.
func (lda *LDA) gibbsSamplingIteration() {
	var wg sync.WaitGroup
	for d := 0; d < lda.totalDocs; d++ {
		wg.Add(1)
		go func(docIndex int) {
			defer wg.Done()
			lda.sampleDocument(docIndex)
		}(d)
	}
	wg.Wait()
}

// sampleDocument samples topics for a single document.
func (lda *LDA) sampleDocument(docIndex int) {
	doc := lda.documents[docIndex]
	for n, wordID := range doc {
		oldTopic := lda.topicAssignments[docIndex][n]

		lda.mutex.Lock()
		lda.docTopicCounts[docIndex][oldTopic]--
		lda.topicWordCounts[oldTopic][wordID]--
		lda.topicCounts[oldTopic]--
		lda.mutex.Unlock()

		distribution := lda.computeTopicDistribution(docIndex, wordID)

		newTopic := sample(distribution)

		lda.mutex.Lock()
		lda.topicAssignments[docIndex][n] = newTopic
		lda.docTopicCounts[docIndex][newTopic]++
		lda.topicWordCounts[newTopic][wordID]++
		lda.topicCounts[newTopic]++
		lda.mutex.Unlock()
	}
}

// computeTopicDistribution computes the topic distribution for a word.
func (lda *LDA) computeTopicDistribution(docIndex int, wordID int) []float64 {
	distribution := make([]float64, lda.numTopics)
	for k := 0; k < lda.numTopics; k++ {
		term1 := (float64(lda.topicWordCounts[k][wordID]) + lda.beta) / (float64(lda.topicCounts[k]) + float64(lda.vocabularySize)*lda.beta)
		term2 := (float64(lda.docTopicCounts[docIndex][k]) + lda.alpha) / (float64(lda.docLengths[docIndex]) + float64(lda.numTopics)*lda.alpha)
		distribution[k] = term1 * term2
	}
	return distribution
}

// sample samples a topic index from the distribution.
func sample(distribution []float64) int {
	sum := 0.0
	for _, val := range distribution {
		sum += val
	}
	r := rand.Float64() * sum
	cumulative := 0.0
	for i, val := range distribution {
		cumulative += val
		if r <= cumulative {
			return i
		}
	}
	return len(distribution) - 1
}

// tokenize splits the text into words.
func tokenize(text string) []string {
	// Simple tokenizer; improve with more sophisticated NLP if needed
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	return tokens
}
