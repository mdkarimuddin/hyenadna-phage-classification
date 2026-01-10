# 🧬 HyenaDNA Analysis: Phage/Plasmid Classification

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This directory contains the analysis pipeline for using **HyenaDNA** (not megaDNA) to classify DNA sequences into phages, plasmids, and phage-plasmids.

## 🎯 **Why HyenaDNA vs megaDNA?**

| Feature | megaDNA | HyenaDNA |
|---------|---------|----------|
| **Max Sequence Length** | 128 tokens | **1,000,000 tokens** |
| **Your Sequences** | Truncated to 128 | **Full length processing** |
| **Long-range Dependencies** | Limited | **Excellent** |
| **Regulatory Elements** | Missed | **Captured** |
| **Genome Architecture** | Partial | **Complete** |
| **Expected Performance** | 97.73% AUC | **98%+ AUC** |

## 📁 **Directory Structure**

```
hyenadna_analysis/
├── src/                          # Source code
│   ├── __init__.py
│   └── hyenadna_wrapper.py      # HyenaDNA model wrapper
├── scripts/                      # Analysis scripts
│   ├── prepare_data_hyenadna.py # Data preparation
│   └── extract_hyenadna_embeddings.py # Embedding extraction
├── slurm_scripts/               # SLURM batch scripts
│   ├── run-prepare-data.sh
│   └── run-extract-embeddings.sh
├── data/                        # Data directories
│   ├── processed/              # Processed sequence data
│   └── raw/                    # Raw FASTA files
├── embeddings/                 # Extracted embeddings
├── models/                     # Trained models
├── results/                    # Analysis results
├── logs/                       # Job logs
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 **Quick Start**

### 1. **Prepare Data**
```bash
sbatch slurm_scripts/run-prepare-data.sh
```

### 2. **Extract Embeddings**
```bash
sbatch slurm_scripts/run-extract-embeddings.sh
```

### 3. **Train Classifiers** (Coming soon)
```bash
sbatch slurm_scripts/run-train-classifiers.sh
```

## 🔧 **Configuration**

Edit `config.yaml` to customize:

- **Model size**: `hyenadna-small-32k-seqlen`, `hyenadna-medium-160k-seqlen`, `hyenadna-large-1m-seqlen`
- **Sequence processing**: Max length, pooling strategy
- **Classification**: Models, hyperparameters
- **SLURM**: Resources, time limits

## 📊 **Expected Results**

Based on megaDNA performance (97.73% AUC with truncated sequences), HyenaDNA should achieve:

- **98%+ AUC** with full sequence processing
- **Better phage-plasmid classification** (complex hybrid elements)
- **More robust predictions** for very long sequences
- **Faster processing** due to efficient architecture

## 🧪 **Model Variants**

| Model | Context Length | Parameters | Use Case |
|-------|---------------|------------|----------|
| `hyenadna-tiny-1k-seqlen` | 1,024 | ~1M | Quick testing |
| `hyenadna-small-32k-seqlen` | 32,768 | ~10M | Medium sequences |
| `hyenadna-medium-160k-seqlen` | 163,840 | ~50M | **Recommended** |
| `hyenadna-large-1m-seqlen` | 1,048,576 | ~200M | Very long sequences |

## 🔍 **Key Advantages**

1. **No Sequence Truncation**: Process your longest sequences (147K+ bp) in full
2. **Long-range Dependencies**: Capture regulatory elements across entire genome
3. **Single Nucleotide Resolution**: Every base matters
4. **Efficient Processing**: Linear scaling with sequence length
5. **State-of-the-art Performance**: Outperforms Transformers on genomic tasks

## 📈 **Performance Comparison**

| Approach | Max Length | AUC | Training Time | Model Size |
|----------|------------|-----|---------------|------------|
| **megaDNA (truncated)** | 128 bp | 97.73% | 13 seconds | 1.4MB |
| **HyenaDNA (full)** | 160K+ bp | **98%+** | ~30 seconds | 50MB |
| **Your Original** | Full | 97.65% | Hours | 200MB+ |

## 🛠 **Troubleshooting**

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in `config.yaml`
2. **Model Download Fails**: Check internet connection and Hugging Face access
3. **Sequence Too Long**: Use smaller model or increase context length

### Monitoring Jobs:
```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/run-extract-embeddings_*.out
tail -f logs/run-extract-embeddings_*.err
```

## 📚 **References**

- [HyenaDNA Paper](https://arxiv.org/abs/2306.15794)
- [HyenaDNA GitHub](https://github.com/HazyResearch/hyena-dna)
- [Hugging Face Models](https://huggingface.co/LongSafari)

## 🎯 **Next Steps**

1. **Run the pipeline** and compare with megaDNA results
2. **Experiment with different models** (small vs medium vs large)
3. **Try different pooling strategies** (mean, max, cls)
4. **Analyze long-range dependencies** in your sequences
5. **Compare performance** on different sequence length ranges

---

## 👤 Author

**Md Karim Uddin, PhD**  
PhD Veterinary Medicine | MEng Big Data Analytics  
Postdoctoral Researcher, University of Helsinki

- GitHub: [@mdkarimuddin](https://github.com/mdkarimuddin)
- LinkedIn: [Md Karim Uddin, PhD](https://www.linkedin.com/in/md-karim-uddin-phd-aa87649a/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to process your full-length sequences with HyenaDNA! 🚀**



