import os
import sys
import re
import math
import time
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# 1. 1,200-CHUNK COMPREHENSIVE MULTI-DOMAIN TECHNICAL CORPUS (4 ARCHETYPES)
# ---------------------------------------------------------------------------

DOC_ARCHETYPES = [
    (
        "Technical_RFC_Specifications.pdf",
        "Dense Technical Specification, RFC Standards & Protocol Manual",
        [
            ("TLS 1.3 CRYPTOGRAPHIC PROTOCOL SPECIFICATION", "Transport Layer Security (TLS) version 1.3, standardized under RFC 8446, is defined as a cryptographic transport security protocol that eliminates legacy ciphers, mandates Diffie-Hellman ephemeral key exchanges for perfect forward secrecy, and completes handshakes in a single round trip."),
            ("WEBSOCKET PROTOCOL FRAMING RFC 6455", "The WebSocket protocol (RFC 6455) is defined as a bidirectional, full-duplex communication protocol operating over a single TCP connection initiated via an HTTP 101 Switching Protocols upgrade request with Sec-WebSocket-Key headers."),
            ("SERVER-SENT EVENTS STREAMING SPECIFICATION", "Server-Sent Events (SSE) is defined as a unidirectional HTTP streaming protocol standardized under W3C HTML5 where a client establishes a persistent connection over text/event-stream and the server pushes UTF-8 events without polling."),
            ("HTTP/3 AND QUIC TRANSPORT FRAMES", "HTTP/3 is defined as the third major version of the Hypertext Transfer Protocol relying on QUIC, a transport layer protocol built over UDP that incorporates native TLS 1.3 encryption, connection migration, and independent stream recovery."),
            ("WRITE-AHEAD LOGGING (WAL) AND ARIES PROTOCOL", "Write-Ahead Logging (WAL) is defined as a durability protocol where transaction log records must be written to append-only disk storage before corresponding dirty buffer pool pages are flushed to disk, guaranteeing ACID recovery via the ARIES protocol."),
            ("B+ TREE BALANCED SEARCH INDEX STRUCTURE", "A B+ Tree is defined as an N-ary balanced search tree data structure where all payload key-value records reside exclusively in doubly linked leaf pages at uniform depth, while interior nodes store routing separator keys."),
            ("LOG-STRUCTURED MERGE-TREE (LSM-TREE) ARCHITECTURE", "A Log-Structured Merge-tree (LSM-tree) is defined as an append-optimized storage engine architecture that buffers incoming writes in an in-memory MemTable before flushing sorted runs to immutable disk SSTables via background compaction."),
            ("AES-256-GCM AUTHENTICATED BLOCK CIPHER", "AES-256-GCM is defined as an authenticated symmetric block cipher combining the Advanced Encryption Standard with Galois/Counter Mode to deliver confidential encryption and integrity authentication in a single algorithmic pass."),
            ("SQLITE CONCURRENCY AND PRAGMA WAL MODE", "SQLite Write-Ahead Logging mode (PRAGMA journal_mode=WAL;) is defined as a database concurrency architecture where concurrent readers access database pages from disk without blocking while a single writer appends transactions to a auxiliary -wal file."),
            ("HMAC-SHA256 SIGNATURE VERIFICATION IN JWT", "JSON Web Tokens (JWT) using HS256 are defined as a compact URL-safe claims representation where the payload is cryptographically authenticated using an HMAC-SHA256 signature calculated over the base64-encoded header and payload with a shared secret.")
        ]
    ),
    (
        "DeepLearning_Research_Papers.pdf",
        "Academic & Research Papers (Mathematical Foundations & Deep Learning)",
        [
            ("TRANSFORMER MULTI-HEAD SELF-ATTENTION", "The Transformer architecture relies on multi-head self-attention mechanisms allowing models to compute dynamic contextual representations across token sequences in parallel without recurrent sequential processing."),
            ("ROTARY POSITION EMBEDDINGS (ROPE) MATHEMATICS", "Rotary Position Embeddings (RoPE) is defined as a positional encoding method that captures relative token distance by applying an orthogonal rotation matrix to query and key representations in complex vector space."),
            ("FLASHATTENTION SRAM TILING ALGORITHM", "FlashAttention is defined as an exact, IO-aware self-attention algorithm that eliminates intermediate attention matrix materialization to HBM by computing softmax tiling within high-speed GPU on-chip SRAM caches."),
            ("MIXTURE OF EXPERTS (MOE) SPARSE ROUTING", "Mixture of Experts (MoE) is defined as a sparse neural architecture where a gating network dynamically routes input tokens to a specialized subset of feed-forward expert sub-networks per forward pass."),
            ("CATASTROPHIC FORGETTING AND EWC REGULARIZATION", "Catastrophic forgetting occurs when sequential fine-tuning on new data degrades previously acquired knowledge. Elastic Weight Consolidation (EWC) penalizes changes to parameters critical for previous tasks through Fisher information matrix regularization."),
            ("KEY-VALUE (KV) CACHE IN AUTOREGRESSIVE DECODING", "The Key-Value (KV) cache is defined as an inference optimization technique that stores intermediate attention keys and values generated during previous autoregressive decoding steps, reducing inference complexity from O(N^2) to O(N)."),
            ("LOW-RANK ADAPTATION (LORA) PARAMETER-EFFICIENT TUNING", "Low-Rank Adaptation (LoRA) is defined as a parameter-efficient fine-tuning technique that freezes base model weights and injects trainable rank-decomposition matrices into transformer self-attention layers."),
            ("TEMPERATURE AND TOP-P NUCLEUS SAMPLING THEORY", "Temperature scaling controls the flatness of the output softmax probability distribution, while Top-p (nucleus) sampling dynamically restricts candidate token generation to the smallest set of tokens whose cumulative probability exceeds p."),
            ("RETRIEVAL-AUGMENTED GENERATION (RAG) PATTERNS", "Retrieval-Augmented Generation (RAG) is defined as an architectural pattern that combines parametric neural language models with non-parametric vector database retrieval to ground generated answers on external verifiable knowledge."),
            ("HALLUCINATION MITIGATION VIA CONTEXT GROUNDING", "Hallucination mitigation in generative AI employs strict system prompt constraints, citation pruning, and retrieval-augmented context grounding to penalize unsubstantiated claims and enforce faithful text generation.")
        ]
    ),
    (
        "Basel_III_and_Regulatory_Compliance.pdf",
        "Legal, Regulatory & Banking Policy Frameworks",
        [
            ("BASEL III CAPITAL ADEQUACY ACCORD", "Basel III is defined as an international regulatory accord developed by the Basel Committee on Banking Supervision that establishes minimum capital adequacy, leverage ratios, and liquidity requirements for global financial institutions."),
            ("COMMON EQUITY TIER 1 (CET1) REGULATORY CAPITAL", "Common Equity Tier 1 (CET1) is defined as the highest quality of regulatory capital under Basel III, comprising common shares, retained earnings, and accumulated other comprehensive income capable of absorbing unexpected losses on a going-concern basis."),
            ("RISK-WEIGHTED ASSETS (RWA) STANDARDIZED APPROACH", "Risk-Weighted Assets (RWA) represent a bank's assets weighted according to credit risk, market risk, and operational risk. Under the Standardized Approach, risk weights range from 0% for sovereign debt to 150% for high-risk corporate exposures."),
            ("LIQUIDITY COVERAGE RATIO (LCR) 30-DAY STRESS BUFFER", "The Liquidity Coverage Ratio (LCR) is defined as the requirement for banks to hold an unencumbered buffer of High-Quality Liquid Assets (HQLA) sufficient to cover total net cash outflows over a 30-day severe stress scenario, maintaining an LCR above 100%."),
            ("NET STABLE FUNDING RATIO (NSFR) STRUCTURAL MATURITY", "The Net Stable Funding Ratio (NSFR) is defined as the ratio of available stable funding (ASF) to required stable funding (RSF) over a one-year horizon, designed to prevent structural maturity mismatches between assets and liabilities."),
            ("FINCEN SUSPICIOUS ACTIVITY REPORT (SAR) REQUIREMENTS", "Financial institutions must file Suspicious Activity Reports (SARs) with regulatory bodies (such as FinCEN) when transactions aggregating $5,000 or more involve known or suspected criminal activity or lack clear economic purpose."),
            ("PCI-DSS 4.0 REQUIREMENT 3.4 DATA ENCRYPTION", "PCI-DSS 4.0 Requirement 3.4 mandates that primary account numbers (PAN) must be rendered unreadable anywhere they are stored using strong one-way cryptographic hashes, truncation, index tokens, or strong AES encryption."),
            ("SOC 2 TYPE 1 AND TYPE 2 AUDIT SCOPE", "SOC 2 Type 1 evaluates the design of security controls at a single point in time, whereas SOC 2 Type 2 assesses the operational effectiveness of trust service criteria (Security, Availability, Confidentiality) over a 6 to 12 month audit window."),
            ("NIST SP 800-207 ZERO TRUST ARCHITECTURE PRINCIPLES", "Zero Trust Architecture (ZTA) is defined as a cybersecurity paradigm codified under NIST SP 800-207 that assumes breaches are inevitable and requires strict continuous authentication, authorization, and cryptographic verification of every access request."),
            ("GDPR DATA SUBJECT ACCESS RIGHTS AND ERASURE", "Under the General Data Protection Regulation (GDPR), Article 17 defines the Right to Erasure (Right to be Forgotten), requiring data controllers to erase personal data without undue delay when data is no longer necessary for original processing purposes.")
        ]
    ),
    (
        "Cloud_Distributed_Systems_Guide.pdf",
        "Narrative & Explanatory Infrastructure Guide (Cloud & Distributed Systems)",
        [
            ("VIRTUAL PRIVATE CLOUD (VPC) NETWORK ISOLATION", "Virtual Private Cloud (VPC) is defined as an isolated virtual network environment in public cloud platforms allowing provisioning of compute resources with custom IP address ranges, subnets, and route tables."),
            ("AWS IDENTITY AND ACCESS MANAGEMENT (IAM) POLICIES", "AWS Identity and Access Management (IAM) refers to the web service for securely controlling authentication and authorization to cloud resources using granular JSON policy documents."),
            ("SECURITY TOKEN SERVICE (STS) ASSUMEROLE DELEGATION", "The AWS Security Token Service (STS) AssumeRole API generates temporary, short-lived security credentials comprising an AccessKeyId, SecretAccessKey, and SessionToken for cross-account delegation."),
            ("CAP THEOREM DISTRIBUTED TRADE-OFFS", "The CAP theorem proves that distributed data stores can provide at most two of three guarantees simultaneously: Consistency, Availability, and Partition tolerance. Eventual consistency provides high availability at the cost of stale reads during network splits."),
            ("RAFT DISTRIBUTED CONSENSUS PROTOCOL", "The Raft consensus algorithm is defined as a distributed consensus protocol that establishes state machine replication through leader election, log replication, and commitment safety invariants across distributed cluster nodes."),
            ("DISASTER RECOVERY RTO AND RPO METRICS", "Recovery Time Objective (RTO) is defined as the maximum acceptable duration of system downtime after an outage, while Recovery Point Objective (RPO) is the maximum acceptable data loss measured in time."),
            ("THUNDERING HERD API STORM MITIGATION", "Thundering herd API storms occur when thousands of concurrent clients simultaneously query an expired cache key. Probabilistic early expiration and randomized exponential backoff with full jitter prevent database overload."),
            ("REDIS REDLOCK DISTRIBUTED LOCK ALGORITHM", "The Redlock algorithm is defined as a distributed mutual exclusion protocol across independent master Redis nodes that acquires leases with monotonic time counters and majority quorum approval."),
            ("CIRCUIT BREAKER FAULT TOLERANCE PATTERN", "The Circuit Breaker pattern is defined as a microservices resilience pattern that prevents cascading failures by monitoring downstream error rates and transitioning between Closed, Open, and Half-Open states to fail fast."),
            ("TOKEN BUCKET RATE LIMITING MECHANICS", "The token bucket rate limiting algorithm is defined as a traffic-shaping algorithm that accumulates tokens at a fixed refill rate up to a burst capacity limit, consuming tokens per request and returning HTTP 429 when exhausted.")
        ]
    )
]

def build_1200_chunk_corpus() -> List[Dict[str, Any]]:
    corpus = []
    chunk_counter = 1

    for doc_name, doc_desc, core_chunks in DOC_ARCHETYPES:
        # 10 canonical definition / ground-truth chunks per document
        for title, text in core_chunks:
            corpus.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "doc": doc_name,
                "archetype": doc_desc,
                "page": (chunk_counter % 60) + 1,
                "text": f"{title}\n{text}"
            })
            chunk_counter += 1

        # 290 operational distractor chunks per document = 300 chunks per document = 1,200 chunks total
        for sub_i in range(1, 291):
            base_idx = (sub_i % len(core_chunks))
            base_title, _ = core_chunks[base_idx]
            tech_kw = base_title.split()[0]
            
            distractor_text = (
                f"{base_title} ENTERPRISE OPERATIONS & TELEMETRY SUITE (MODULE {sub_i})\n"
                f"Operational deployment of {tech_kw} components in high-throughput enterprise infrastructure requires "
                f"continuous runtime telemetry. Systems administrators must configure connection pools, thread quotas, "
                f"and audit logging pipelines according to enterprise compliance baselines. For operational parameter set {sub_i}, "
                f"the supervisory daemon monitors p99 latency percentiles, memory buffer exhaustion, and worker thread contention. "
                f"In the event of hardware degradation or threshold violation, traffic is routed to passive standby replicas."
            )
            corpus.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "doc": doc_name,
                "archetype": doc_desc,
                "page": ((chunk_counter + sub_i) % 60) + 1,
                "text": distractor_text
            })
            chunk_counter += 1

    return corpus

print(f"Building 1,200-chunk corpus...")
corpus = build_1200_chunk_corpus()
print(f"Total Chunks Ingested: {len(corpus)}")
