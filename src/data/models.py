"""
SQLAlchemy ORM models for the stock predictor database.
"""

from datetime import datetime, date

from sqlalchemy import (
    Column, Integer, BigInteger, String, Numeric,
    Date, DateTime, ForeignKey, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship

from src.data.database import Base


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector = Column(String(100))
    currency = Column(String(10), default="BRL")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prices = relationship("StockPrice", back_populates="asset", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="asset", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="asset", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Asset(ticker='{self.ticker}')>"


class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_stock_prices_asset_date"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4))
    adj_close = Column(Numeric(12, 4))
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    asset = relationship("Asset", back_populates="prices")

    def __repr__(self):
        return f"<StockPrice(asset_id={self.asset_id}, date='{self.date}', close={self.close})>"


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint("asset_id", "date", "feature_name", name="uq_features_asset_date_name"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Numeric(18, 8))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    asset = relationship("Asset", back_populates="features")

    def __repr__(self):
        return f"<Feature(asset_id={self.asset_id}, date='{self.date}', name='{self.feature_name}')>"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    prediction_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    model_version = Column(String(50), nullable=False)
    predicted_direction = Column(Integer)
    predicted_return = Column(Numeric(10, 6))
    confidence = Column(Numeric(5, 4))
    actual_direction = Column(Integer)
    actual_return = Column(Numeric(10, 6))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    asset = relationship("Asset", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(asset_id={self.asset_id}, target='{self.target_date}', direction={self.predicted_direction})>"


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    ticker = Column(String(20), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Numeric(10, 6))
    n_train_samples = Column(Integer)
    n_test_samples = Column(Integer)
    hyperparameters = Column(JSON)

    def __repr__(self):
        return f"<ModelMetric(version='{self.model_version}', metric='{self.metric_name}', value={self.metric_value})>"
