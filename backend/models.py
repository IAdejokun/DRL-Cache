# backend/models.py - setup the tables 
from sqlalchemy import Column, BigInteger, Integer, Boolean, Text, TIMESTAMP, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db import Base

class CacheItem(Base):
    __tablename__ = "cache_items"
    id = Column(BigInteger, primary_key=True, index=True)
    object_id = Column(Text, unique=True, nullable=False, index=True)
    size_bytes = Column(BigInteger, nullable=False)
    last_updated_ts = Column(TIMESTAMP(timezone=True), nullable=False)
    ttl_s = Column(Integer, nullable=False)
    inserted_ts = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

class Request(Base):
    __tablename__ = "requests"
    id = Column(BigInteger, primary_key=True, index=True)
    ts = Column(TIMESTAMP(timezone=True), nullable=False)
    client_id = Column(Text)
    object_id = Column(Text, nullable=False, index=True)
    object_size_bytes = Column(BigInteger, nullable=False)
    origin_latency_ms = Column(Integer, nullable=False)
    was_write = Column(Boolean, nullable=False, server_default="false")

    outcome = relationship("Outcome", back_populates="request", uselist=False)

class Outcome(Base):
    __tablename__ = "outcomes"
    id = Column(BigInteger, primary_key=True, index=True)
    request_id = Column(BigInteger, ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True)
    cache_hit = Column(Boolean, nullable=False)
    served_latency_ms = Column(Integer, nullable=False)
    staleness_s = Column(Integer, nullable=False, server_default="0")

    request = relationship("Request", back_populates="outcome")

class AgentEvent(Base):
    __tablename__ = "agent_events"
    id = Column(BigInteger, primary_key=True, index=True)
    ts = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True)
    state_json = Column(JSON, nullable=False)
    action = Column(Text, nullable=False)
    reward = Column(BigInteger)  # keep simple; can switch to Float later
    object_id = Column(Text, index=True)
  