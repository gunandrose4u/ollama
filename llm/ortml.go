package llm

import (
	"io"
)

type ortModel struct {
	*containerORT
	kv      KV
	tensors []*Tensor
}

// KV implements model.
func (llm *ortModel) KV() KV {
	return llm.kv
}

// Tensors implements model.
func (llm *ortModel) Tensors() Tensors {
	return llm.tensors
}

// NumCtx implements model.
func (*ortModel) NumCtx() uint32 {
	return 0
}

// NumEmbed implements model.
func (*ortModel) NumEmbed() uint32 {
	return 0
}

// NumGQA implements model.
func (*ortModel) NumGQA() uint32 {
	return 0
}

// NumHead implements model.
func (*ortModel) NumHead() uint32 {
	return 0
}

// NumHeadKv implements model.
func (*ortModel) NumHeadKv() uint32 {
	return 0
}

// NumLayers implements model.
func (*ortModel) NumLayers() uint32 {
	return 0
}

func newOrtModel(container *containerORT) *ortModel {
	return &ortModel{
		containerORT: container,
		kv:           make(KV),
	}
}

func (llm *ortModel) ModelFamily() string {
	return "ort_genai_0.18.0"
}

func (llm *ortModel) ModelType() string {
	return "transformers"
}

func (llm *ortModel) FileType() string {
	return "onnx"
}

func (c *containerORT) Name() string {
	return "ort"
}

func (c *containerORT) Decode(rso io.ReadSeeker) (model, error) {
	model := newOrtModel(c)
	rso.Seek(0, io.SeekEnd)
	return model, nil
}
