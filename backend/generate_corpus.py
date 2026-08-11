import os
import sys
import json
import re
import math
import time
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# 1. GENERATE 900-CHUNK COMPREHENSIVE MULTI-DOMAIN TECHNICAL CORPUS
# ---------------------------------------------------------------------------

def generate_large_corpus() -> List[Dict[str, Any]]:
    corpus = []
    chunk_counter = 1

    domains = [
        ("AI_Transformers_DeepLearning.pdf", "AI Foundations & Deep Learning", [
            ("INTRODUCTION TO ARTIFICIAL INTELLIGENCE", "Artificial Intelligence (AI), commonly referred to as AI, is a branch of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."),
            ("MACHINE LEARNING FOUNDATIONS", "Machine Learning is a subset of artificial intelligence that focuses on building algorithms that learn patterns from empirical training data rather than relying exclusively on explicitly programmed rule-based instructions."),
            ("SUPERVISED LEARNING METHODS", "Supervised learning, also known as directed learning, refers to algorithms trained on input-output pairs where ground-truth labels guide loss minimization via gradient descent."),
            ("UNSUPERVISED LEARNING TECHNIQUES", "Unsupervised learning is the general term for algorithms that uncover latent representations, clusters, or probability distributions from unlabeled datasets without explicit target supervision."),
            ("REINFORCEMENT LEARNING ARCHITECTURE", "Reinforcement Learning (RL) is defined as a framework where autonomous agents learn optimal decision-making policies through trial-and-error interactions with an environment, maximizing cumulative scalar rewards."),
            ("RETRIEVAL-AUGMENTED GENERATION (RAG)", "Retrieval-Augmented Generation (RAG) is defined as an architectural pattern that combines parametric neural language models with non-parametric vector database retrieval to ground generated answers on external verifiable knowledge."),
            ("TRANSFORMER ATTENTION MECHANISMS", "The Transformer architecture relies on multi-head self-attention mechanisms allowing models to compute dynamic contextual representations across token sequences in parallel without recurrent sequential processing."),
            ("ROTARY POSITION EMBEDDINGS (ROPE)", "Rotary Position Embeddings (RoPE) is a relatively new form of positional encoding that encodes relative position information by rotating query and key representations in complex vector space."),
            ("FLASHATTENTION GPU KERNEL OPTIMIZATION", "FlashAttention is an exact, fast, memory-efficient self-attention algorithm that reduces GPU memory access overhead by computing softmax tiling within high-speed SRAM caches."),
            ("MIXTURE OF EXPERTS (MOE)", "Mixture of Experts (MoE) is defined as a sparse neural architecture where a gating network dynamically routes input tokens to a specialized subset of feed-forward expert sub-networks per forward pass."),
            ("CATASTROPHIC FORGETTING AND REGULARIZATION", "Catastrophic forgetting occurs when sequential fine-tuning on new data degrades previously acquired knowledge. Elastic Weight Consolidation (EWC) penalizes changes to parameters critical for previous tasks through Fisher information matrix regularization."),
            ("KV CACHE OPTIMIZATION IN LLM INFERENCE", "The Key-Value (KV) cache stores intermediate attention keys and values generated during previous autoregressive decoding steps, eliminating redundant matrix multiplications and reducing inference latency from O(N^2) to O(N)."),
            ("LOW-RANK ADAPTATION (LORA)", "Low-Rank Adaptation (LoRA) is defined as a parameter-efficient fine-tuning technique that freezes base model weights and injects trainable rank-decomposition matrices into transformer self-attention layers."),
            ("TEMPERATURE AND TOP-P NUCLEUS SAMPLING", "Temperature scaling controls the flatness of the output softmax probability distribution, while Top-p (nucleus) sampling dynamically restricts candidate token generation to the smallest set of tokens whose cumulative probability exceeds p."),
            ("HALLUCINATION MITIGATION VIA GROUNDING", "Hallucination mitigation in generative AI employs strict system prompt constraints, citation pruning, and retrieval-augmented context grounding to penalize unsubstantiated claims and enforce faithful text generation.")
        ]),
        ("Cloud_AWS_Distributed_Systems.pdf", "Cloud Architecture & Distributed Infrastructure", [
            ("VIRTUAL PRIVATE CLOUD (VPC) ISOLATION", "Virtual Private Cloud (VPC) is defined as an isolated virtual network environment in public cloud platforms allowing provisioning of compute resources with custom IP address ranges, subnets, and route tables."),
            ("AWS IDENTITY AND ACCESS MANAGEMENT (IAM)", "AWS Identity and Access Management (IAM) refers to the web service for securely controlling authentication and authorization to cloud resources using granular JSON policy documents."),
            ("SECURITY TOKEN SERVICE (STS) ASSUMEROLE", "The AWS Security Token Service (STS) AssumeRole API generates temporary, short-lived security credentials comprising an AccessKeyId, SecretAccessKey, and SessionToken for cross-account delegation."),
            ("S3 STORAGE CLASSES AND LIFECYCLE TIERS", "Amazon S3 offers multiple storage classes including S3 Standard, S3 Standard-IA, S3 Glacier Flexible Retrieval, and S3 Glacier Deep Archive with configurable automated lifecycle transition policies."),
            ("EC2 AUTO SCALING AND TARGET TRACKING", "EC2 Auto Scaling dynamically adjusts the number of compute instances in an Auto Scaling Group based on metrics like CPU utilization or custom CloudWatch metrics using target tracking scaling policies."),
            ("ELASTIC LOAD BALANCING (ALB VS NLB)", "Application Load Balancer (ALB) operates at Layer 7 supporting path-based routing, while Network Load Balancer (NLB) operates at Layer 4 handling ultra-high throughput with ultra-low latencies and static IP support."),
            ("DYNAMODB PARTITIONING AND GLOBAL SECONDARY INDEXES", "Amazon DynamoDB achieves horizontal scalability by partitioning data across SSD storage nodes using partition key hashing. Global Secondary Indexes (GSI) enable queries on non-primary key attributes."),
            ("CAP THEOREM AND CONSISTENCY MODELS", "The CAP theorem proves that distributed data stores can provide at most two of three guarantees simultaneously: Consistency, Availability, and Partition tolerance. Eventual consistency provides high availability at the cost of stale reads during network splits."),
            ("RAFT CONSENSUS PROTOCOL IN DISTRIBUTED SYSTEMS", "The Raft consensus algorithm establishes distributed state machine replication through leader election, log replication, and commitment safety invariants across distributed cluster nodes."),
            ("DISASTER RECOVERY RTO AND RPO METRICS", "Recovery Time Objective (RTO) is defined as the maximum acceptable duration of system downtime after an outage, while Recovery Point Objective (RPO) is the maximum acceptable data loss measured in time."),
            ("CROSS-REGION DATABASE REPLICATION", "Cross-region asynchronous replication replicates database write transactions across distant geographic regions, enabling global read scalability and disaster recovery failover within minutes."),
            ("THUNDERING HERD API MITIGATION", "Thundering herd API storms occur when thousands of concurrent clients simultaneously query an expired cache key. Probabilistic early expiration and randomized exponential backoff with full jitter prevent database overload."),
            ("DISTRIBUTED LOCKING WITH REDIS REDLOCK", "The Redlock algorithm establishes distributed mutual exclusion across independent master Redis nodes by acquiring leases with monotonic time counters and quorum approval."),
            ("AWS KMS ENVELOPE ENCRYPTION", "AWS Key Management Service (KMS) implements envelope encryption where a Customer Master Key (CMK) encrypts a plaintext Data Key (DK), which in turn encrypts local data files."),
            ("CLOUD METRICS AND OPENTELEMETRY TRACING", "Distributed tracing with OpenTelemetry propagates trace IDs and span contexts across microservices over HTTP headers, enabling end-to-end latency profiling and bottleneck isolation.")
        ]),
        ("Cybersecurity_ZeroTrust_Compliance.pdf", "Cybersecurity, Cryptography & Compliance", [
            ("ZERO TRUST ARCHITECTURE (ZTA)", "Zero Trust Architecture (ZTA) is defined as a cybersecurity paradigm that assumes breaches are inevitable and requires strict continuous authentication, authorization, and cryptographic verification of every user and device access request, codified under NIST SP 800-207."),
            ("AES-256-GCM AUTHENTICATED ENCRYPTION", "AES-256-GCM is an authenticated symmetric block cipher combining Advanced Encryption Standard with Galois/Counter Mode to provide both confidential data encryption and cryptographic message authentication in a single pass."),
            ("TLS 1.3 CRYPTOGRAPHIC PROTOCOL", "Transport Layer Security (TLS) version 1.3 (RFC 8446) eliminates legacy cryptographic algorithms, mandates Diffie-Hellman ephemeral key exchanges for perfect forward secrecy, and reduces handshake latency to a single round trip."),
            ("PUBLIC KEY INFRASTRUCTURE (PKI) AND OCSP", "Public Key Infrastructure (PKI) manages digital certificates via Certificate Authorities (CAs). Online Certificate Status Protocol (OCSP) stapling allows web servers to provide timestamped, signed certificate validity proofs directly to clients."),
            ("OAUTH 2.0 AND OPENID CONNECT (OIDC)", "OAuth 2.0 is an authorization framework enabling third-party applications to obtain scoped access via bearer tokens, while OpenID Connect (OIDC) extends OAuth 2.0 with JSON Web Tokens (JWT) for user identity authentication."),
            ("JSON WEB TOKEN (JWT) SECURITY INVARIANTS", "JSON Web Tokens (JWT) encapsulate JSON claims signed with HMAC-SHA256 (HS256) or RSA-SHA256 (RS256). Secure implementations must strictly reject the 'none' algorithm and validate signature signatures against a server secret."),
            ("BCRYPT PASSWORD HASHING WORK FACTOR", "Bcrypt is a cryptographic password hashing function based on the Blowfish cipher incorporating unique 128-bit salts and an adaptive work factor (iteration count) to resist FPGA and GPU brute-force attacks."),
            ("CROSS-SITE SCRIPTING (XSS) DEFENSE", "Cross-Site Scripting (XSS) vulnerabilities occur when untrusted user input is rendered into HTML DOM contexts without context-aware escaping. Content Security Policy (CSP) headers and framework-level JSX escaping mitigate reflected and stored XSS."),
            ("SQL INJECTION MITIGATION AND PARAMETERIZATION", "SQL Injection vulnerabilities occur when dynamic SQL queries concatenate unescaped input strings. Complete protection requires parameterized prepared statements with positional bind parameters (e.g. '?' or '$1')."),
            ("CROSS-ORIGIN RESOURCE SHARING (CORS)", "Cross-Origin Resource Sharing (CORS) enforces HTTP header policies allowing servers to specify which origin domains are permitted to load protected resources via Access-Control-Allow-Origin."),
            ("SOC 2 TYPE 1 AND TYPE 2 COMPLIANCE", "SOC 2 Type 1 evaluates the design of security controls at a single point in time, whereas SOC 2 Type 2 assesses the operational effectiveness of trust service criteria (Security, Availability, Confidentiality) over a 6 to 12 month audit window."),
            ("NIST SP 800-61 INCIDENT RESPONSE LIFECYCLE", "The NIST incident response lifecycle comprises four sequential phases: Preparation, Detection and Analysis, Containment Eradication and Recovery, and Post-Incident Activity (Lessons Learned)."),
            ("ROLE-BASED VS ATTRIBUTE-BASED ACCESS CONTROL", "Role-Based Access Control (RBAC) assigns permissions to predefined roles, while Attribute-Based Access Control (ABAC) evaluates dynamic contextual attributes (user role, resource sensitivity, IP address, time) using policy engines."),
            ("SUPPLY CHAIN SECURITY AND SLSA FRAMEWORK", "Supply-chain Levels for Software Artifacts (SLSA) provides a security framework with verifiable build provenance, non-falsifiable metadata, and isolated hermetic build environments to prevent software tampering."),
            ("MULTI-FACTOR AUTHENTICATION (MFA) AND FIDO2", "FIDO2 WebAuthn provides phishing-resistant multi-factor authentication using asymmetric public-key cryptography bound to hardware security authenticators (YubiKeys, TPMs).")
        ]),
        ("Financial_Risk_Compliance_Basel_III.pdf", "Financial Risk Management & Banking Compliance", [
            ("BASEL III CAPITAL ADEQUACY FRAMEWORK", "Basel III is defined as an international regulatory accord developed by the Basel Committee on Banking Supervision that establishes minimum capital adequacy, leverage ratios, and liquidity requirements for global financial institutions."),
            ("COMMON EQUITY TIER 1 (CET1) CAPITAL", "Common Equity Tier 1 (CET1) is the highest quality of regulatory capital under Basel III, comprising common shares, retained earnings, and accumulated other comprehensive income capable of absorbing unexpected losses on a going-concern basis."),
            ("RISK-WEIGHTED ASSETS (RWA) CALCULATION", "Risk-Weighted Assets (RWA) represent a bank's assets weighted according to credit risk, market risk, and operational risk. Under the Standardized Approach, risk weights range from 0% for sovereign debt to 150% for high-risk corporate exposures."),
            ("LIQUIDITY COVERAGE RATIO (LCR) REQUIREMENT", "The Liquidity Coverage Ratio (LCR) requires banks to hold an unencumbered buffer of High-Quality Liquid Assets (HQLA) sufficient to cover total net cash outflows over a 30-day severe stress scenario, maintaining an LCR above 100%."),
            ("NET STABLE FUNDING RATIO (NSFR)", "The Net Stable Funding Ratio (NSFR) is defined as the ratio of available stable funding (ASF) to required stable funding (RSF) over a one-year horizon, designed to prevent structural maturity mismatches between assets and liabilities."),
            ("ANTI-MONEY LAUNDERING (AML) AND KYC", "Anti-Money Laundering (AML) regulations mandate customer due diligence (CDD), beneficial ownership verification, and continuous transaction monitoring to prevent illicit funds from entering the financial system."),
            ("SUSPICIOUS ACTIVITY REPORT (SAR) THRESHOLDS", "Financial institutions must file Suspicious Activity Reports (SARs) with regulatory bodies (such as FinCEN) when transactions aggregating $5,000 or more involve known or suspected criminal activity or lack clear economic purpose."),
            ("CHARGEBACK DISPUTE LIFECYCLE (VISA & MASTERCARD)", "The payment dispute lifecycle proceeds from retrieval request to first chargeback, issuer representment of compelling evidence, pre-arbitration review, and formal card brand network arbitration."),
            ("PCI-DSS 4.0 REQUIREMENT 3.4 DATA PROTECTION", "PCI-DSS 4.0 Requirement 3.4 mandates that primary account numbers (PAN) must be rendered unreadable anywhere they are stored using strong one-way cryptographic hashes, truncation, index tokens, or strong AES encryption."),
            ("REAL-TIME TRANSACTION FRAUD DETECTION", "Automated transaction fraud engines evaluate velocity counters, device fingerprints, geolocation inconsistencies, and machine learning anomaly scores within sub-100 millisecond authorization windows."),
            ("CREDIT DEFAULT SWAPS (CDS) AND SPREAD RISK", "A Credit Default Swap (CDS) is a financial derivative that transfers the credit exposure of fixed-income products between parties, where the buyer pays periodic premiums to insure against reference entity default."),
            ("VALUE AT RISK (VAR) AND EXPECTED SHORTFALL", "Value at Risk (VaR) measures the maximum potential loss over a specified time horizon at a given confidence level (e.g. 99% 10-day VaR). Expected Shortfall (ES) measures the average loss in the tail beyond the VaR threshold."),
            ("STRESS TESTING AND CCAR REGULATION", "The Comprehensive Capital Analysis and Review (CCAR) is an annual regulatory exercise by the Federal Reserve assessing whether large bank holding companies have sufficient capital under severely adverse macroeconomic stress scenarios."),
            ("SANCTIONS SCREENING AND PEP MATCHING", "Automated sanctions screening matches customer names against OFAC Specially Designated Nationals (SDN) lists and Politically Exposed Persons (PEP) databases using fuzzy phonetic string algorithms."),
            ("MERCHANT ACQUIRING SETTLEMENT AND INTERCHANGE", "Payment card interchange fees represent fees paid between acquiring banks and issuing banks for card transaction processing, governed by interchange-plus pricing models.")
        ]),
        ("Database_Internals_and_Storage_Engines.pdf", "Database Systems & Storage Engine Internals", [
            ("WRITE-AHEAD LOGGING (WAL) AND ARIES RECOVERY", "Write-Ahead Logging (WAL) is defined as a durability technique where transaction modifications are recorded in append-only log files on non-volatile storage before corresponding database dirty pages are flushed to disk, ensuring ACID atomicity via ARIES recovery protocol."),
            ("B+ TREE INDEX STRUCTURE AND PAGE SPLITS", "A B+ Tree is an N-ary balanced search tree where all data records are stored in doubly linked leaf nodes at uniform depth, while internal nodes store routing keys. Node splits occur when page occupancy exceeds capacity (e.g. 4096 bytes)."),
            ("LSM-TREE (LOG-STRUCTURED MERGE-TREE)", "A Log-Structured Merge-tree (LSM-tree) is defined as an append-optimized storage engine architecture that buffers writes in an in-memory MemTable before flushing sorted runs to immutable disk SSTables, using background compaction."),
            ("MULTI-VERSION CONCURRENCY CONTROL (MVCC)", "Multi-Version Concurrency Control (MVCC) enables concurrent reads and writes without locking by maintaining multiple physical row versions tagged with creation and deletion transaction timestamps (XMIN/XMAX)."),
            ("TWO-PHASE LOCKING (2PL) AND DEADLOCKS", "Strict Two-Phase Locking (SS2PL) guarantees serializability by acquiring shared locks for reads and exclusive locks for writes, holding all exclusive locks until transaction commit to prevent cascading aborts."),
            ("TWO-PHASE COMMIT (2PC) DISTRIBUTED TRANSACTIONS", "Two-Phase Commit (2PC) coordinates atomic transaction commits across distributed database nodes through a Prepare phase (voting) followed by a Commit or Abort phase managed by a coordinator node."),
            ("DATABASE NORMALIZATION (1NF TO BCNF)", "Database normalization organizes relational schemas to eliminate anomalies. Boyce-Codd Normal Form (BCNF) requires that for every non-trivial functional dependency X -> Y, X must be a superkey."),
            ("QUERY OPTIMIZER AND COST ESTIMATION", "Relational query optimizers evaluate candidate physical execution plans using cost models based on CPU instruction counts, I/O page reads, and histogram-based selectivity estimations."),
            ("SQLITE MEMORY-MAPPED I/O AND WAL MODE", "SQLite WAL mode (`PRAGMA journal_mode=WAL;`) allows concurrent readers to proceed uninterrupted while a single writer appends transactions to a `-wal` auxiliary file without blocking."),
            ("COLUMN-ORIENTED STORAGE ENGINES", "Columnar databases store data by attribute column rather than record row, maximizing compression efficiency via dictionary encoding and bit-packing while accelerating OLAP aggregation queries."),
            ("BITMAP INDEX SCANS AND ROARING BITMAPS", "Bitmap indexing represents distinct attribute values as binary bit vectors. Roaring Bitmaps optimize sparse bitmap storage by partitioning keys into 16-bit chunk containers."),
            ("PAGE BUFFER POOL MANAGEMENT (LRU-K)", "The database buffer pool caches disk pages in RAM using replacement policies like LRU-K, which tracks the backward distance to the K-th previous page reference to resist scan thrashing."),
            ("DATABASE REPLICATION LAG AND READ REPLICAS", "Replication lag measures the time delay for written master transactions to be replayed on asynchronous read replicas. Read-your-own-writes consistency requires routing subsequent reads to the primary until replicas catch up."),
            ("CHECKPOINTING AND DIRTY PAGE FLUSHING", "Database checkpointing periodically syncs in-memory dirty buffer pool pages to disk and truncates the write-ahead log, bounding crash recovery restart times."),
            ("VECTOR SIMILARITY SEARCH AND FAISS INDEXING", "FAISS (Facebook AI Similarity Search) is defined as a library for efficient dense vector nearest neighbor indexing and search, providing exact IndexFlatL2 distance scans and approximate Inverted File (IVF) quantization.")
        ]),
        ("Modern_Web_Protocols_and_APIs.pdf", "Networking Protocols & Modern Web Architectures", [
            ("SERVER-SENT EVENTS (SSE) STREAMING", "Server-Sent Events (SSE) is defined as a unidirectional HTTP streaming protocol standardized under HTML5 where a client establishes a persistent connection over `text/event-stream` and the server pushes UTF-8 events without client polling."),
            ("WEBSOCKET RFC 6455 DUPLEX PROTOCOL", "The WebSocket protocol (RFC 6455) provides full-duplex bidirectional communication channels over a single TCP connection, initialized via an HTTP 101 Switching Protocols upgrade handshake."),
            ("HTTP/2 MULTIPLEXING AND BINARY FRAMING", "HTTP/2 introduces a binary framing layer that multiplexes multiple concurrent request/response streams over a single TCP connection, eliminating head-of-line blocking at the application layer."),
            ("HTTP/3 AND QUIC UDP-BASED TRANSPORT", "HTTP/3 replaces TCP with QUIC, a transport protocol built on top of UDP that incorporates integrated TLS 1.3 encryption, connection migration across IP changes, and independent stream loss recovery."),
            ("CROSS-ORIGIN RESOURCE SHARING (CORS) PREFLIGHT", "CORS preflight requests use the HTTP `OPTIONS` method with headers `Origin`, `Access-Control-Request-Method`, and `Access-Control-Request-Headers` to verify server cross-origin permissions before executing non-simple requests."),
            ("TOKEN BUCKET AND LEAKY BUCKET RATE LIMITING", "The token bucket rate limiting algorithm accumulates tokens at a fixed refill rate up to a burst capacity limit, consuming tokens per request and returning HTTP 429 Too Many Requests when exhausted."),
            ("REST API IDEMPOTENCY AND IDEMPOTENCY-KEY", "Idempotent HTTP methods (GET, PUT, DELETE) produce identical server state when executed multiple times. Non-idempotent POST requests achieve safety using client-provided `Idempotency-Key` headers stored in Redis."),
            ("GRAPHQL VS RESTFUL API ARCHITECTURES", "GraphQL provides declarative data fetching where clients request exact field schemas in a single round trip, preventing over-fetching and under-fetching at the cost of complex query parsing and caching challenges."),
            ("SERVERLESS COMPUTE AND COLD START OVERHEAD", "Serverless functions (e.g. AWS Lambda) execute in ephemeral microVM containers. Cold start latency occurs when a new container runtime is initialized on demand after periods of inactivity."),
            ("CIRCUIT BREAKER PATTERN IN MICROSERVICES", "The Circuit Breaker pattern prevents cascading failures in distributed systems by monitoring downstream error rates and transitioning between Closed, Open, and Half-Open states to fail fast without overloading degraded dependencies."),
            ("API GATEWAY PATTERNS AND REVERSE PROXIES", "An API Gateway serves as a unified entrypoint for microservices, handling cross-cutting concerns including SSL termination, JWT authentication, rate limiting, request transformation, and path routing."),
            ("DNS RESOLUTION AND ANYCAST ROUTING", "Domain Name System (DNS) resolves human-readable domain names into IP addresses. Anycast routing advertises identical IP prefixes from multiple global BGP locations, routing client packets to the topologically closest datacenter."),
            ("CDN EDGE CACHING AND CACHE-CONTROL HEADERS", "Content Delivery Networks (CDNs) cache static assets and API responses at edge locations based on `Cache-Control: public, max-age=3600, s-maxage=86400, stale-while-revalidate` directives."),
            ("TCP 3-WAY HANDSHAKE AND SLOW START", "TCP establishes reliable connections via SYN, SYN-ACK, ACK 3-way handshakes. Congestion control starts in Slow Start, exponentially doubling the congestion window (CWND) every round-trip time until packet loss occurs."),
            ("COOKIE SECURITY ATTRIBUTES (SAMESITE & SECURE)", "HTTP cookies mitigating session hijacking must include the `HttpOnly` flag (preventing JavaScript DOM access), `Secure` flag (enforcing HTTPS transmission), and `SameSite=Strict` or `Lax` (preventing CSRF exploitation).")
        ])
    ]

    for doc_name, domain_title, core_chunks in domains:
        # Generate 150 chunks per domain = 900 chunks total
        # Base 15 core chunks + 135 detailed operational / distractor chunks
        for title, text in core_chunks:
            corpus.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "doc": doc_name,
                "page": (chunk_counter % 50) + 1,
                "text": f"{title}\n{text}"
            })
            chunk_counter += 1

        # Generate realistic operational / distraction chunks where technical terms are used in applied context
        # (e.g., configuring ZTA, tuning B+ Tree page sizes, deploying WAL databases) without containing the definitional clause
        for sub_i in range(1, 136):
            base_idx = (sub_i % len(core_chunks))
            base_title, _ = core_chunks[base_idx]
            tech_keywords = base_title.split()[0]
            
            op_text = (
                f"{base_title} IMPLEMENTATION & OPERATIONAL PROCEDURES (PART {sub_i})\n"
                f"When deploying and administering {tech_keywords} infrastructure in enterprise production environments, "
                f"engineers must configure telemetry metrics, connection pooling, buffer utilization, and error recovery policies. "
                f"Specifically, system parameter {sub_i} requires monitoring latency percentiles (p95, p99), validating CPU thread contention, "
                f"and maintaining audit logs conforming to organizational compliance requirements. In case of operational threshold breaches, "
                f"the supervisory control loop initiates failover routines to secondary replicas and alerts on-call site reliability engineers."
            )
            corpus.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "doc": doc_name,
                "page": ((chunk_counter + sub_i) % 50) + 1,
                "text": op_text
            })
            chunk_counter += 1

    return corpus

print(f"Generating large realistic multi-domain corpus...")
corpus = generate_large_corpus()
print(f"Total Chunks Generated: {len(corpus)}")
