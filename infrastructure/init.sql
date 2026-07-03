-- ============================================
-- Stock Trend Predictor - Database Schema
-- ============================================

-- Tabela de ativos (metadados do ticker)
CREATE TABLE IF NOT EXISTS assets (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255),
    sector VARCHAR(100),
    currency VARCHAR(10) DEFAULT 'BRL',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tabela principal: dados OHLCV diários
CREATE TABLE IF NOT EXISTS stock_prices (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    adj_close NUMERIC(12, 4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(asset_id, date)
);

-- Índice para consultas por data (séries temporais)
CREATE INDEX idx_stock_prices_date ON stock_prices(date);
CREATE INDEX idx_stock_prices_asset_date ON stock_prices(asset_id, date DESC);

-- Tabela de features calculadas
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value NUMERIC(18, 8),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(asset_id, date, feature_name)
);

CREATE INDEX idx_features_asset_date ON features(asset_id, date DESC);

-- Tabela de predições do modelo
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_direction INTEGER,  -- 1 = alta, 0 = baixa
    predicted_return NUMERIC(10, 6),
    confidence NUMERIC(5, 4),
    actual_direction INTEGER,
    actual_return NUMERIC(10, 6),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_asset_target ON predictions(asset_id, target_date DESC);

-- Tabela de métricas de treinamento
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    trained_at TIMESTAMP DEFAULT NOW(),
    ticker VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC(10, 6),
    n_train_samples INTEGER,
    n_test_samples INTEGER,
    hyperparameters JSONB
);

-- Função para atualizar updated_at automaticamente
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_assets_updated_at
    BEFORE UPDATE ON assets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
