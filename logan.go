package logan

import (
	"math/rand"
	"time"
	"encoding/json"
)

type Model struct {
	Weights []float64  `json:"weights"`
	Bias    float64    `json:"bias"`
	Eta     float64    `json:"learningRate"`
	rand    *rand.Rand
}

func New(learningRate float64) *Model {
	return &Model {
		Weights: []float64 {},
		Bias: 0,
		Eta: learningRate,
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
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
