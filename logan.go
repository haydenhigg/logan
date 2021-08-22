package logan

import (
	"math/rand"
	"time"
	"encoding/json"
)

type Model struct {
	Weights      []float64 `json:"weights"`
	Bias         float64   `json:"bias"`
	LearningRate float64   `json:"learningRate"`
	L2Parameter  float64   `json:"l2Parameter"`
	Means        []float64 `json:"means"`
	rand         *rand.Rand
}

type Gradient struct {
	weights []float64
	bias    float64
}

func NewL2Regularized(learningRate, l2Parameter float64) *Model {
	return &Model {
		LearningRate: learningRate,
		L2Parameter: l2Parameter,
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func New(learningRate float64) *Model {
	return NewL2Regularized(learningRate, 0)
}

func Marshal(mdl *Model) ([]byte, error) {
	return json.Marshal(mdl)
}

func Unmarshal(data []byte) (*Model, error) {
	var mdl Model

	if err := json.Unmarshal(data, &mdl); err == nil {
		mdl.rand = rand.New(rand.NewSource(time.Now().UnixNano()))
		return &mdl, nil
	} else {
		return nil, err
	}
}
