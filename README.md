```text
main()
│
├── cublasCreate()
├── model.load()
│   ├── fread(...) x5         <-- Loads constants: C, E, D, H, O
│   ├── tra[i].load()         <-- Transformer blocks
│   │   ├── a.load()
│   │   ├── b.load()
│   │   └── c.load()
│   └── out.load()
│
├── model.generate(input, n)
│   ├── s.push_back(...)                <-- Initial char append
│   ├── memset(vs, 0)
│   ├── loop: for each char
│   │   └── sample(x, p)
│   │       ├── emb.data = _wyhash64()  <-- Initializes input embedding
│   │       ├── tra[d].fw(...)          <-- Transformer forward
│   │       │   ├── a.fw()
│   │       │   │   ├── u.fw()
│   │       │   │   │   ├── linear::fw()
│   │       │   │   │   │   ├── _s16() (kernel)
│   │       │   │   │   │   ├── cublasTSSgemvStridedBatched()
│   │       │   │   │   │   └── _l16() (kernel)
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── _selffsuv() (kernel)
│   │       │   │   └── o.fw()
│   │       │   ├── b.fw()
│   │       │   │   ├── x.fw()
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── _s16() x3 (kernel)
│   │       │   │   ├── cublasTSSgemvStridedBatched() x2
│   │       │   │   ├── _sexyfp() (kernel)
│   │       │   │   ├── _sexyfsuv() (kernel)
│   │       │   │   ├── _layernorm() (kernel)
│   │       │   │   ├── o.fw()
│   │       │   │   └── _sexyadd() (kernel)
│   │       │   └── c.fw()
│   │       │       ├── u.fw()
│   │       │       │   └── linear::fw() (same as above)
│   │       │       ├── _layernorm() (kernel)
│   │       │       ├── _selffsuv() (kernel)
│   │       │       ├── o.fw()
│   │       │       └── _sexyadd() (kernel)
│   │       ├── _layernorm() (kernel)
│   │       └── out.fw()
│   │           └── linear::fw()
│   ├── cudaDeviceSynchronize()
│   ├── softmax()
│   ├── wy2u01(wyrand(&prng))   <-- Sampling next token
│   └── s.push_back(c)          <-- Append next token
│
├── gettimeofday() x2
└── cublasDestroy()

Miscellaneous
├── _wyhash64(A, B)            <-- Custom 64-bit hash
│   └── _wymum(A, B)
├── wy2u01(r)                  <-- Int → Float conversion [0,1)
├── wyrand(&seed)             <-- Pseudo-random number gen
├── _s16() / _l16()           <-- Float ↔ bfloat16 (CUDA kernels)
├── _layernorm()              <-- Standard transformer normalization (CUDA)
├── _sexyfp(), _sexyfsuv(), _sexyadd(), _selffsuv()  <-- Custom fused ops
```