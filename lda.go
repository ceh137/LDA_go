package LDA_go

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type LDA struct {
	numTopics      int
	alpha          float64
	eta            float64
	documents      [][]int
	vocabulary     []string
	word2id        map[string]int
	id2word        map[int]string
	vocabularySize int

	lambda      [][]float64 // Topic-word distributions
	gamma       [][]float64 // Document-topic distributions
	updateCount int

	mutex sync.Mutex
}

// NewLDA creates a new LDA model with the specified number of topics.
// Default values for alpha and eta are set to 1/numTopics and 1/numTopics respectively.
func NewLDA(numTopics int) *LDA {
	alpha := 1.0 / float64(numTopics)
	eta := 1.0 / float64(numTopics)
	return &LDA{
		numTopics: numTopics,
		alpha:     alpha,
		eta:       eta,
		lambda:    make([][]float64, numTopics),
	}
}

// TrainFromFile trains the LDA model using data from a file.
func (lda *LDA) TrainFromFile(filename string, numIterations int) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	documents := strings.Split(string(data), "\n")
	return lda.Train(documents, numIterations)
}

// UpdateFromFile updates the LDA model incrementally using data from a file.
func (lda *LDA) UpdateFromFile(filename string, numIterations int) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	documents := strings.Split(string(data), "\n")
	return lda.Update(documents, numIterations)
}

// Train trains the LDA model on the provided documents.
func (lda *LDA) Train(documents []string, numIterations int) error {
	if len(documents) == 0 {
		return errors.New("no documents provided for training")
	}

	lda.buildVocabulary(documents)
	lda.initializeLambda()

	for iter := 0; iter < numIterations; iter++ {
		err := lda.onlineUpdate(documents)
		if err != nil {
			return err
		}
		// Optional: Add logging to monitor convergence
		fmt.Printf("Iteration %d completed.\n", iter+1)
	}

	return nil
}

// Update incrementally updates the LDA model with new documents.
func (lda *LDA) Update(documents []string, numIterations int) error {
	lda.extendVocabulary(documents)

	for iter := 0; iter < numIterations; iter++ {
		err := lda.onlineUpdate(documents)
		if err != nil {
			return err
		}
		// Optional: Add logging to monitor convergence
		fmt.Printf("Update iteration %d completed.\n", iter+1)
	}

	return nil
}

// GetTopics returns the top N words for each topic.
func (lda *LDA) GetTopics(numWords int) [][]string {
	topics := make([][]string, lda.numTopics)
	for k := 0; k < lda.numTopics; k++ {
		wordProbabilities := make([]wordProbability, lda.vocabularySize)
		for w := 0; w < lda.vocabularySize; w++ {
			prob := lda.lambda[k][w]
			wordProbabilities[w] = wordProbability{
				word:        lda.id2word[w],
				probability: prob,
			}
		}
		sort.Slice(wordProbabilities, func(i, j int) bool {
			return wordProbabilities[i].probability > wordProbabilities[j].probability
		})
		topWords := make([]string, numWords)
		for i := 0; i < numWords && i < len(wordProbabilities); i++ {
			topWords[i] = wordProbabilities[i].word
		}
		topics[k] = topWords
	}
	return topics
}

// SaveModel saves the model weights to a file in JSON format.
func (lda *LDA) SaveModel(filename string) error {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	data := map[string]interface{}{
		"lambda":      lda.lambda,
		"vocabulary":  lda.vocabulary,
		"word2id":     lda.word2id,
		"id2word":     lda.id2word,
		"numTopics":   lda.numTopics,
		"alpha":       lda.alpha,
		"eta":         lda.eta,
		"updateCount": lda.updateCount,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(filename, jsonData, 0644)
}

// LoadModel loads the model weights from a file.
func (lda *LDA) LoadModel(filename string) error {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	data := make(map[string]interface{})
	err = json.Unmarshal(jsonData, &data)
	if err != nil {
		return err
	}

	lda.numTopics = int(data["numTopics"].(float64))
	lda.alpha = data["alpha"].(float64)
	lda.eta = data["eta"].(float64)
	lda.updateCount = int(data["updateCount"].(float64))

	// Load vocabulary and mappings
	vocabInterface := data["vocabulary"].([]interface{})
	lda.vocabulary = make([]string, len(vocabInterface))
	for i, word := range vocabInterface {
		lda.vocabulary[i] = word.(string)
	}
	lda.vocabularySize = len(lda.vocabulary)

	word2idInterface := data["word2id"].(map[string]interface{})
	lda.word2id = make(map[string]int)
	for k, v := range word2idInterface {
		lda.word2id[k] = int(v.(float64))
	}

	id2wordInterface := data["id2word"].(map[string]interface{})
	lda.id2word = make(map[int]string)
	for k, v := range id2wordInterface {
		idx, _ := strconv.Atoi(k)
		lda.id2word[idx] = v.(string)
	}

	// Load lambda
	lambdaInterface := data["lambda"].([]interface{})
	lda.lambda = make([][]float64, lda.numTopics)
	for k, topicInterface := range lambdaInterface {
		topicSlice := topicInterface.([]interface{})
		lda.lambda[k] = make([]float64, lda.vocabularySize)
		for w, val := range topicSlice {
			lda.lambda[k][w] = val.(float64)
		}
	}

	return nil
}

// ResetModel deletes existing model weights and starts from scratch.
func (lda *LDA) ResetModel() error {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	lda.lambda = make([][]float64, lda.numTopics)
	lda.vocabulary = nil
	lda.word2id = nil
	lda.id2word = nil
	lda.vocabularySize = 0
	lda.updateCount = 0

	return nil
}

// internal functions

func (lda *LDA) initializeLambda() {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	seededRand := rand.New(rand.NewSource(time.Now().UnixNano()))
	for k := 0; k < lda.numTopics; k++ {
		lda.lambda[k] = make([]float64, lda.vocabularySize)
		for w := 0; w < lda.vocabularySize; w++ {
			lda.lambda[k][w] = seededRand.Float64() + 1e-2
		}
		normalize(lda.lambda[k])
	}
}

func (lda *LDA) onlineUpdate(documents []string) error {
	// Convert documents to word IDs
	docs := make([][]int, len(documents))
	for i, doc := range documents {
		words := tokenize(doc)
		wordIDs := make([]int, 0, len(words))
		for _, word := range words {
			if id, ok := lda.word2id[word]; ok {
				wordIDs = append(wordIDs, id)
			}
		}
		docs[i] = wordIDs
	}

	var wg sync.WaitGroup
	gammaUpdates := make([][]float64, len(docs))
	for i := range docs {
		wg.Add(1)
		go func(d int) {
			defer wg.Done()
			gamma, err := lda.variationalInference(docs[d])
			if err != nil {
				fmt.Printf("Error in variational inference: %v\n", err)
			}
			gammaUpdates[d] = gamma
		}(i)
	}
	wg.Wait()

	lda.mutex.Lock()
	defer lda.mutex.Unlock()
	rho := math.Pow(float64(lda.updateCount)+float64(1), -0.7) // Learning rate
	for d, doc := range docs {
		gamma := gammaUpdates[d]
		phi := make([][]float64, len(doc))
		for n, wordID := range doc {
			phi_nk := make([]float64, lda.numTopics)
			sum := 0.0
			for k := 0; k < lda.numTopics; k++ {
				phi_nk[k] = lda.lambda[k][wordID] * math.Exp(digamma(gamma[k]))
				sum += phi_nk[k]
			}
			for k := 0; k < lda.numTopics; k++ {
				phi_nk[k] /= sum
			}
			phi[n] = phi_nk
		}
		// Update lambda
		for k := 0; k < lda.numTopics; k++ {
			for n, wordID := range doc {
				lda.lambda[k][wordID] = (1-rho)*lda.lambda[k][wordID] + rho*(lda.eta+float64(len(docs))*phi[n][k])
			}
			normalize(lda.lambda[k])
		}
	}
	lda.updateCount++
	return nil
}

func (lda *LDA) variationalInference(doc []int) ([]float64, error) {
	gamma := make([]float64, lda.numTopics)
	for k := 0; k < lda.numTopics; k++ {
		gamma[k] = lda.alpha + float64(len(doc))/float64(lda.numTopics)
	}

	phi := make([][]float64, len(doc))
	maxIter := 50
	for iter := 0; iter < maxIter; iter++ {
		gammaOld := make([]float64, lda.numTopics)
		copy(gammaOld, gamma)
		for n, wordID := range doc {
			phi_nk := make([]float64, lda.numTopics)
			sum := 0.0
			for k := 0; k < lda.numTopics; k++ {
				phi_nk[k] = lda.lambda[k][wordID] * math.Exp(digamma(gamma[k]))
				sum += phi_nk[k]
			}
			for k := 0; k < lda.numTopics; k++ {
				phi_nk[k] /= sum
			}
			phi[n] = phi_nk
		}
		for k := 0; k < lda.numTopics; k++ {
			gamma[k] = lda.alpha
			for n := 0; n < len(doc); n++ {
				gamma[k] += phi[n][k]
			}
		}
		if converged(gamma, gammaOld, 1e-6) {
			break
		}
	}
	return gamma, nil
}

func (lda *LDA) buildVocabulary(documents []string) {
	wordSet := make(map[string]struct{})
	for _, doc := range documents {
		words := tokenize(doc)
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
}

func (lda *LDA) extendVocabulary(documents []string) {
	wordSet := make(map[string]struct{})
	for _, word := range lda.vocabulary {
		wordSet[word] = struct{}{}
	}
	newWords := 0
	for _, doc := range documents {
		words := tokenize(doc)
		for _, word := range words {
			if _, exists := wordSet[word]; !exists {
				wordSet[word] = struct{}{}
				lda.vocabulary = append(lda.vocabulary, word)
				lda.word2id[word] = lda.vocabularySize
				lda.id2word[lda.vocabularySize] = word
				lda.vocabularySize++
				newWords++
			}
		}
	}
	if newWords > 0 {
		// Extend lambda
		for k := 0; k < lda.numTopics; k++ {
			lda.lambda[k] = append(lda.lambda[k], make([]float64, newWords)...)
			normalize(lda.lambda[k])
		}
	}
}

type wordProbability struct {
	word        string
	probability float64
}

// Helper functions

func tokenize(text string) []string {
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	return tokens
}

func normalize(vec []float64) {
	sum := 0.0
	for _, val := range vec {
		sum += val
	}
	for i := range vec {
		vec[i] /= sum
	}
}

func digamma(x float64) float64 {
	// Simple approximation of digamma function
	result := 0.0
	for x < 7 {
		result -= 1 / x
		x++
	}
	x -= 1.0 / 2.0
	xx := 1.0 / x
	xx2 := xx * xx
	xx4 := xx2 * xx2
	result += math.Log(x) + (1.0/24.0)*xx2 - (7.0/960.0)*xx4
	return result
}

func converged(a, b []float64, tol float64) bool {
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}
