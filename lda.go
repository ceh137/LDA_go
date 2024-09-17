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

// LDA represents the Latent Dirichlet Allocation model.
type LDA struct {
	NumTopics      int
	Alpha          float64
	Eta            float64
	Vocabulary     []string
	Word2id        map[string]int
	Id2word        map[int]string
	VocabularySize int

	Lambda      [][]float64 // Topic-word distributions
	UpdateCount int

	mutex sync.RWMutex
}

// NewLDA creates a new LDA model with the specified number of topics.
func NewLDA(NumTopics int) *LDA {
	Alpha := 1.0 / float64(NumTopics)
	Eta := 1.0 / float64(NumTopics)
	return &LDA{
		NumTopics: NumTopics,
		Alpha:     Alpha,
		Eta:       Eta,
		Lambda:    make([][]float64, NumTopics),
	}
}

// TrainFromFile trains the LDA model using data from a file.
// It now supports iteration passing via channels and error handling.
func (lda *LDA) TrainFromFile(filename string, numIterations int, iterCh chan int, errCh chan error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		errCh <- err
		close(iterCh)
		close(errCh)
		return
	}
	documents := strings.Split(string(data), "\n")
	lda.Train(documents, numIterations, iterCh, errCh)
}

// UpdateFromFile updates the LDA model incrementally using data from a file.
func (lda *LDA) UpdateFromFile(filename string, numIterations int, iterCh chan int, errCh chan error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		errCh <- err
		close(iterCh)
		close(errCh)
		return
	}
	documents := strings.Split(string(data), "\n")
	lda.Update(documents, numIterations, iterCh, errCh)
}

// Train trains the LDA model on the provided documents.
func (lda *LDA) Train(documents []string, numIterations int, iterCh chan int, errCh chan error) {
	if len(documents) == 0 {
		errCh <- errors.New("no documents provided for training")
		close(iterCh)
		close(errCh)
		return
	}

	lda.buildVocabulary(documents)
	lda.initializeLambda()

	go func() {
		for iter := 0; iter < numIterations; iter++ {
			err := lda.onlineUpdate(documents)
			if err != nil {
				errCh <- err
				break
			}
			iterCh <- iter + 1
		}
		close(iterCh)
		close(errCh)
	}()
}

// Update incrementally updates the LDA model with new documents.
func (lda *LDA) Update(documents []string, numIterations int, iterCh chan int, errCh chan error) {
	lda.extendVocabulary(documents)

	go func() {
		for iter := 0; iter < numIterations; iter++ {
			err := lda.onlineUpdate(documents)
			if err != nil {
				errCh <- err
				break
			}
			iterCh <- iter + 1
		}
		close(iterCh)
		close(errCh)
	}()
}

// GetTopics returns the top N words for each topic.
func (lda *LDA) GetTopics(numWords int) [][]string {
	topics := make([][]string, lda.NumTopics)
	for k := 0; k < lda.NumTopics; k++ {
		wordProbabilities := make([]wordProbability, lda.VocabularySize)
		for w := 0; w < lda.VocabularySize; w++ {
			lda.mutex.RLock()
			prob := lda.Lambda[k][w]
			wordProbabilities[w] = wordProbability{
				word:        lda.Id2word[w],
				probability: prob,
			}
			lda.mutex.RUnlock()
		}
		sort.Slice(wordProbabilities, func(i, j int) bool {
			return wordProbabilities[i].probability > wordProbabilities[j].probability
		})
		topWords := make([]string, 0, numWords)
		for i := 0; i < numWords && i < len(wordProbabilities); i++ {
			topWords = append(topWords, wordProbabilities[i].word)
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
		"Lambda":      lda.Lambda,
		"Vocabulary":  lda.Vocabulary,
		"Word2id":     lda.Word2id,
		"Id2word":     lda.Id2word,
		"NumTopics":   lda.NumTopics,
		"Alpha":       lda.Alpha,
		"Eta":         lda.Eta,
		"UpdateCount": lda.UpdateCount,
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

	lda.NumTopics = int(data["NumTopics"].(float64))
	lda.Alpha = data["Alpha"].(float64)
	lda.Eta = data["Eta"].(float64)
	lda.UpdateCount = int(data["UpdateCount"].(float64))

	// Load Vocabulary and mappings
	vocabInterface := data["Vocabulary"].([]interface{})
	lda.Vocabulary = make([]string, len(vocabInterface))
	for i, word := range vocabInterface {
		lda.Vocabulary[i] = word.(string)
	}
	lda.VocabularySize = len(lda.Vocabulary)

	Word2idInterface := data["Word2id"].(map[string]interface{})
	lda.Word2id = make(map[string]int)
	for k, v := range Word2idInterface {
		lda.Word2id[k] = int(v.(float64))
	}

	Id2wordInterface := data["Id2word"].(map[string]interface{})
	lda.Id2word = make(map[int]string)
	for k, v := range Id2wordInterface {
		idx, _ := strconv.Atoi(k)
		lda.Id2word[idx] = v.(string)
	}

	// Load Lambda
	LambdaInterface := data["Lambda"].([]interface{})
	lda.Lambda = make([][]float64, lda.NumTopics)
	for k, topicInterface := range LambdaInterface {
		topicSlice := topicInterface.([]interface{})
		lda.Lambda[k] = make([]float64, lda.VocabularySize)
		for w, val := range topicSlice {
			lda.Lambda[k][w] = val.(float64)
		}
	}

	return nil
}

// ResetModel deletes existing model weights and starts from scratch.
func (lda *LDA) ResetModel() error {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	lda.Lambda = make([][]float64, lda.NumTopics)
	lda.Vocabulary = nil
	lda.Word2id = nil
	lda.Id2word = nil
	lda.VocabularySize = 0
	lda.UpdateCount = 0

	return nil
}

// Internal functions

func (lda *LDA) initializeLambda() {
	lda.mutex.Lock()
	defer lda.mutex.Unlock()

	seededRand := rand.New(rand.NewSource(time.Now().UnixNano()))
	for k := 0; k < lda.NumTopics; k++ {
		lda.Lambda[k] = make([]float64, lda.VocabularySize)
		for w := 0; w < lda.VocabularySize; w++ {
			lda.Lambda[k][w] = seededRand.Float64() + 1e-2
		}
		normalize(&lda.Lambda[k])
	}
}

func (lda *LDA) onlineUpdate(documents []string) error {
	// Convert documents to word IDs
	docs := make([][]int, len(documents))
	for i, doc := range documents {
		words := Tokenize(doc)
		wordIDs := make([]int, 0, len(words))
		for _, word := range words {
			lda.mutex.Lock()
			if id, ok := lda.Word2id[word]; ok {
				wordIDs = append(wordIDs, id)
			}
			lda.mutex.Unlock()
		}
		docs[i] = wordIDs
	}

	var wg sync.WaitGroup
	gammaUpdates := make([][]float64, len(docs))
	for i := range docs {
		wg.Add(1)
		go func(d int) {
			defer wg.Done()
			gamma, err := lda.VariationalInference(docs[d])
			if err != nil {
				fmt.Printf("Error in variational inference: %v\n", err)
			}
			gammaUpdates[d] = gamma
		}(i)
	}
	wg.Wait()

	lda.mutex.Lock()
	defer lda.mutex.Unlock()
	rho := math.Pow(float64(lda.UpdateCount)+1, -0.7) // Learning rate
	epsilon := 1e-10
	for d, doc := range docs {
		gamma := gammaUpdates[d]
		phi := make([][]float64, len(doc))
		for n, wordID := range doc {
			phi_nk := make([]float64, lda.NumTopics)
			sum := 0.0
			for k := 0; k < lda.NumTopics; k++ {
				gammaK := gamma[k]
				if gammaK <= 0 {
					gammaK = epsilon
				}
				phi_nk[k] = lda.Lambda[k][wordID] * math.Exp(digamma(gammaK))
				phi_nk[k] = checkValid(phi_nk[k])
				sum += phi_nk[k]
			}
			if sum == 0 {
				sum = epsilon
			}
			for k := 0; k < lda.NumTopics; k++ {
				phi_nk[k] /= sum
				phi_nk[k] = checkValid(phi_nk[k])
			}
			phi[n] = phi_nk
		}
		// Update Lambda
		for k := 0; k < lda.NumTopics; k++ {
			for n, wordID := range doc {
				lda.Lambda[k][wordID] = (1-rho)*lda.Lambda[k][wordID] + rho*(lda.Eta+float64(len(docs))*phi[n][k])
				lda.Lambda[k][wordID] = checkValid(lda.Lambda[k][wordID])
			}
			normalize(&lda.Lambda[k])
		}
	}
	lda.UpdateCount++
	return nil
}

func (lda *LDA) VariationalInference(doc []int) ([]float64, error) {
	gamma := make([]float64, lda.NumTopics)
	for k := 0; k < lda.NumTopics; k++ {
		gamma[k] = lda.Alpha + float64(len(doc))/float64(lda.NumTopics)
		if gamma[k] <= 0 {
			gamma[k] = 1e-10
		}
	}

	phi := make([][]float64, len(doc))
	maxIter := 50
	epsilon := 1e-10

	for iter := 0; iter < maxIter; iter++ {
		gammaOld := make([]float64, lda.NumTopics)
		copy(gammaOld, gamma)
		for n, wordID := range doc {
			phi_nk := make([]float64, lda.NumTopics)
			sum := 0.0
			for k := 0; k < lda.NumTopics; k++ {
				gammaK := gamma[k]
				if gammaK <= 0 {
					gammaK = epsilon
				}
				lda.mutex.RLock()
				phi_nk[k] = lda.Lambda[k][wordID] * math.Exp(digamma(gammaK))
				lda.mutex.RUnlock()
				phi_nk[k] = checkValid(phi_nk[k])
				sum += phi_nk[k]
			}
			if sum == 0 {
				sum = epsilon
			}
			for k := 0; k < lda.NumTopics; k++ {
				phi_nk[k] /= sum
				phi_nk[k] = checkValid(phi_nk[k])
			}
			phi[n] = phi_nk
		}
		for k := 0; k < lda.NumTopics; k++ {
			gamma[k] = lda.Alpha
			for n := 0; n < len(doc); n++ {
				gamma[k] += phi[n][k]
			}
			if gamma[k] <= 0 || math.IsNaN(gamma[k]) || math.IsInf(gamma[k], 0) {
				gamma[k] = epsilon
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
		words := Tokenize(doc)
		for _, word := range words {
			wordSet[word] = struct{}{}
		}
	}

	lda.Vocabulary = make([]string, 0, len(wordSet))
	lda.Word2id = make(map[string]int)
	lda.Id2word = make(map[int]string)
	id := 0
	for word := range wordSet {
		lda.Vocabulary = append(lda.Vocabulary, word)
		lda.Word2id[word] = id
		lda.Id2word[id] = word
		id++
	}
	lda.VocabularySize = len(lda.Vocabulary)
}

func (lda *LDA) extendVocabulary(documents []string) {
	wordSet := make(map[string]struct{})
	for _, word := range lda.Vocabulary {
		wordSet[word] = struct{}{}
	}
	newWords := 0
	for _, doc := range documents {
		words := Tokenize(doc)
		for _, word := range words {
			lda.mutex.Lock()
			if _, exists := wordSet[word]; !exists {
				wordSet[word] = struct{}{}
				lda.Vocabulary = append(lda.Vocabulary, word)
				lda.Word2id[word] = lda.VocabularySize
				lda.Id2word[lda.VocabularySize] = word
				lda.VocabularySize++
				newWords++
			}
			lda.mutex.Unlock()
		}
	}
	if newWords > 0 {
		// Extend Lambda
		for k := 0; k < lda.NumTopics; k++ {
			newEntries := make([]float64, newWords)
			for i := 0; i < newWords; i++ {
				newEntries[i] = lda.Eta
			}
			lda.mutex.Lock()
			lda.Lambda[k] = append(lda.Lambda[k], newEntries...)
			normalize(&lda.Lambda[k])
			lda.mutex.Unlock()
		}
	}
}

type wordProbability struct {
	word        string
	probability float64
}

// Helper functions

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	return tokens
}

func normalize(vec *[]float64) {
	sum := 0.0
	for _, val := range *vec {
		sum += val
	}
	if sum == 0 {
		sum = 1e-10 // Small epsilon to prevent division by zero
	}
	for i := range *vec {
		(*vec)[i] /= sum
	}
}

func digamma(x float64) float64 {
	if x <= 0 {
		x = 1e-10
	}
	// Simple approximation of digamma function
	return math.Log(x) - 1/(2*x)
}

func converged(a, b []float64, tol float64) bool {
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func checkValid(value float64) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 1e-10
	}
	return value
}
