-- ============================================
-- MEDALLION ARCHITECTURE - Bronze / Silver / Gold
-- ============================================
-- Bronze: dados brutos exatamente como vieram da fonte
-- Silver: dados limpos, deduplicados, tipados corretamente
-- Gold: dados agregados, features prontas para consumo (ML, BI)

-- ============================================
-- SCHEMAS
-- ============================================
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- ============================================
-- BRONZE LAYER - Raw ingestion (dados crus)
-- ============================================
-- Mantém o dado exatamente como chegou da API, com metadados de ingestão.
-- Nunca é alterado após inserção (append-only / immutable).

CREATE TABLE IF NOT EXISTS bronze.raw_stock_prices (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(14, 6),
    high NUMERIC(14, 6),
    low NUMERIC(14, 6),
    close NUMERIC(14, 6),
    adj_close NUMERIC(14, 6),
    volume BIGINT,
    -- Metadados de ingestão
    source VARCHAR(50) DEFAULT 'yfinance',
    ingested_at TIMESTAMP DEFAULT NOW(),
    ingestion_batch_id VARCHAR(36),  -- UUID do batch de ingestão
    UNIQUE(ticker, date, source)
);

CREATE INDEX idx_bronze_ticker_date ON bronze.raw_stock_prices(ticker, date DESC);
CREATE INDEX idx_bronze_ingested_at ON bronze.raw_stock_prices(ingested_at);

-- Log de ingestão (auditoria)
CREATE TABLE IF NOT EXISTS bronze.ingestion_log (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(36) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    start_date DATE,
    end_date DATE,
    rows_ingested INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL,  -- 'success', 'failed', 'partial'
    error_message TEXT,
    source VARCHAR(50) DEFAULT 'yfinance',
    started_at TIMESTAMP DEFAULT NOW(),
    finished_at TIMESTAMP
);

-- ============================================
-- SILVER LAYER - Cleaned & validated
-- ============================================
-- Dados limpos, deduplicados, com tratamento de missing values,
-- validação de integridade, e tipagem correta.

CREATE TABLE IF NOT EXISTS silver.stock_prices (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open NUMERIC(12, 4) NOT NULL,
    high NUMERIC(12, 4) NOT NULL,
    low NUMERIC(12, 4) NOT NULL,
    close NUMERIC(12, 4) NOT NULL,
    adj_close NUMERIC(12, 4),
    volume BIGINT NOT NULL,
    -- Retornos calculados
    return_simple NUMERIC(10, 8),
    return_log NUMERIC(10, 8),
    -- Data quality flags
    is_imputed BOOLEAN DEFAULT FALSE,
    quality_score NUMERIC(3, 2) DEFAULT 1.00,  -- 0.0 a 1.0
    -- Metadados
    processed_at TIMESTAMP DEFAULT NOW(),
    source_batch_id VARCHAR(36),  -- Referência ao batch da bronze
    UNIQUE(asset_id, date)
);

CREATE INDEX idx_silver_asset_date ON silver.stock_prices(asset_id, date DESC);
CREATE INDEX idx_silver_date ON silver.stock_prices(date);

-- Tabela de qualidade de dados
CREATE TABLE IF NOT EXISTS silver.data_quality_checks (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id),
    check_date DATE NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    passed BOOLEAN NOT NULL,
    details JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- GOLD LAYER - Feature store & ML-ready
-- ============================================
-- Features agregadas, indicadores técnicos, dados prontos para
-- alimentar modelos de ML e dashboards de BI.

CREATE TABLE IF NOT EXISTS gold.feature_store (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    -- Price-based features
    sma_5 NUMERIC(12, 4),
    sma_10 NUMERIC(12, 4),
    sma_21 NUMERIC(12, 4),
    sma_63 NUMERIC(12, 4),
    ema_12 NUMERIC(12, 4),
    ema_26 NUMERIC(12, 4),
    -- Volatility
    volatility_21d NUMERIC(10, 6),
    volatility_63d NUMERIC(10, 6),
    atr_14 NUMERIC(12, 4),
    -- Momentum indicators
    rsi_14 NUMERIC(6, 2),
    rsi_7 NUMERIC(6, 2),
    macd NUMERIC(12, 6),
    macd_signal NUMERIC(12, 6),
    macd_histogram NUMERIC(12, 6),
    -- Bollinger Bands
    bb_upper NUMERIC(12, 4),
    bb_lower NUMERIC(12, 4),
    bb_width NUMERIC(10, 6),
    bb_position NUMERIC(6, 4),
    -- Volume features
    volume_sma_21 NUMERIC(18, 2),
    volume_ratio_21 NUMERIC(8, 4),
    obv NUMERIC(18, 2),
    -- Return features
    return_1d NUMERIC(10, 8),
    return_5d NUMERIC(10, 8),
    return_21d NUMERIC(10, 8),
    -- Lag features
    close_lag_1 NUMERIC(12, 4),
    close_lag_5 NUMERIC(12, 4),
    close_lag_21 NUMERIC(12, 4),
    -- Target
    target_direction_5d INTEGER,  -- 1 = up, 0 = down
    future_return_5d NUMERIC(10, 8),
    -- Metadata
    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(asset_id, date)
);

CREATE INDEX idx_gold_feature_store_asset_date ON gold.feature_store(asset_id, date DESC);
CREATE INDEX idx_gold_feature_store_date ON gold.feature_store(date);

-- Tabela de predições (camada gold - consumível)
CREATE TABLE IF NOT EXISTS gold.predictions (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_direction INTEGER,
    predicted_return NUMERIC(10, 6),
    confidence NUMERIC(5, 4),
    -- Resultado real (preenchido após target_date)
    actual_direction INTEGER,
    actual_return NUMERIC(10, 6),
    is_correct BOOLEAN,
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    evaluated_at TIMESTAMP
);

CREATE INDEX idx_gold_predictions_asset ON gold.predictions(asset_id, target_date DESC);

-- Métricas agregadas do modelo por período
CREATE TABLE IF NOT EXISTS gold.model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    n_predictions INTEGER,
    accuracy NUMERIC(5, 4),
    precision_score NUMERIC(5, 4),
    recall_score NUMERIC(5, 4),
    f1_score NUMERIC(5, 4),
    directional_accuracy NUMERIC(5, 4),
    sharpe_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(8, 4),
    computed_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- VIEWS para facilitar consumo
-- ============================================

-- View: último dado disponível por ativo
CREATE OR REPLACE VIEW gold.latest_prices AS
SELECT DISTINCT ON (a.ticker)
    a.ticker,
    a.name,
    sp.date,
    sp.close,
    sp.volume,
    sp.return_simple AS daily_return
FROM silver.stock_prices sp
JOIN assets a ON sp.asset_id = a.id
ORDER BY a.ticker, sp.date DESC;

-- View: features mais recentes por ativo (para inferência)
CREATE OR REPLACE VIEW gold.latest_features AS
SELECT DISTINCT ON (a.ticker)
    a.ticker,
    fs.*
FROM gold.feature_store fs
JOIN assets a ON fs.asset_id = a.id
ORDER BY a.ticker, fs.date DESC;
