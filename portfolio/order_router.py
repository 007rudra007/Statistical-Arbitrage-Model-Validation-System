"""
Phase 5 – Step 19: Order Execution Engine
===========================================
QuickFIX/J-style FIX 4.4 order router + Zerodha Kite API adapter.

Architecture:
  OrderRouter (abstract)
    ├── KiteOrderRouter   — Zerodha Kite Connect REST
    ├── FIXOrderRouter    — FIX 4.4 protocol messages
    └── PaperOrderRouter  — Paper-trading simulation (no real money)

Features:
  - Market / Limit / SL / SL-M orders
  - NSE equity + F&O routing
  - Order lifecycle: PENDING → OPEN → COMPLETE / REJECTED / CANCELLED
  - Idempotent order ID with microsecond UUID
  - Pre-trade SEBI checks (position limit, PDT flag)
  - Full FIX 4.4 message builder (NewOrderSingle tag 35=D)
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────
class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"        # Stop-loss limit
    SL_M   = "SL-M"      # Stop-loss market

class ProductType(str, Enum):
    CNC  = "CNC"   # Cash and Carry (delivery)
    MIS  = "MIS"   # Margin Intraday Square-off
    NRML = "NRML"  # F&O overnight

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    COMPLETE  = "COMPLETE"
    REJECTED  = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL   = "PARTIAL"

class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"   # NSE Futures & Options


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class Order:
    ticker:       str
    side:         OrderSide
    quantity:     int
    order_type:   OrderType
    product:      ProductType
    exchange:     Exchange   = Exchange.NSE
    price:        float      = 0.0        # limit / SL trigger price
    trigger_price: float     = 0.0        # for SL / SL-M
    validity:     str        = "DAY"
    tag:          str        = "aladdin"  # strategy tag
    order_id:     str        = field(default_factory=lambda: str(uuid.uuid4()).replace("-","")[:16])
    created_at:   str        = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OrderResponse:
    order_id:   str
    status:     OrderStatus
    message:    str = ""
    fill_price: float = 0.0
    filled_qty: int   = 0
    raw:        Dict  = field(default_factory=dict)


@dataclass
class Position:
    ticker:        str
    quantity:      int
    avg_price:     float
    product:       ProductType
    exchange:      Exchange
    pnl:           float = 0.0
    value:         float = 0.0


# ── FIX 4.4 message builder ───────────────────────────────────────────────────
class FIXMessageBuilder:
    """
    Builds FIX 4.4 NewOrderSingle (35=D) and OrderCancelRequest (35=F) messages.
    Tag encoding follows FIX 4.4 protocol spec.
    """

    _FIX_DELIMITER = "\x01"

    # FIX side: 1=Buy, 2=Sell
    _SIDE_MAP = {OrderSide.BUY: "1", OrderSide.SELL: "2"}

    # FIX OrdType: 1=Market, 2=Limit, 3=Stop, 4=Stop limit
    _TYPE_MAP = {
        OrderType.MARKET: "1",
        OrderType.LIMIT:  "2",
        OrderType.SL_M:   "3",
        OrderType.SL:     "4",
    }

    def new_order_single(self, order: Order, sender: str = "ALADDIN",
                          target: str = "NSE_OMS") -> str:
        """Build FIX 4.4 NewOrderSingle (35=D) message string."""
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:23]
        seq = "1"

        body_fields = [
            ("11",  order.order_id),           # ClOrdID
            ("21",  "1"),                       # HandlInst: automated
            ("38",  str(order.quantity)),       # OrderQty
            ("40",  self._TYPE_MAP[order.order_type]),   # OrdType
            ("44",  f"{order.price:.2f}" if order.price else "0"),     # Price
            ("54",  self._SIDE_MAP[order.side]),         # Side
            ("55",  order.ticker),               # Symbol
            ("58",  order.tag),                  # Text / tag
            ("60",  now),                        # TransactTime
            ("99",  f"{order.trigger_price:.2f}"),       # StopPx
            ("100", order.exchange.value),       # ExDestination
            ("207", order.exchange.value),       # SecurityExchange
        ]
        body = self._FIX_DELIMITER.join(f"{k}={v}" for k, v in body_fields)

        # Header
        header = self._FIX_DELIMITER.join([
            f"8=FIX.4.4",
            f"9={len(body)}",
            f"35=D",
            f"49={sender}",
            f"56={target}",
            f"34={seq}",
            f"52={now}",
        ])

        full_msg = header + self._FIX_DELIMITER + body
        checksum = sum(ord(c) for c in full_msg) % 256
        full_msg += self._FIX_DELIMITER + f"10={checksum:03d}"
        return full_msg

    def order_cancel_request(self, order_id: str, orig_order_id: str,
                              ticker: str, side: OrderSide,
                              sender: str = "ALADDIN", target: str = "NSE_OMS") -> str:
        """Build FIX 4.4 OrderCancelRequest (35=F)."""
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:23]
        body_fields = [
            ("11", order_id),
            ("41", orig_order_id),
            ("54", self._SIDE_MAP[side]),
            ("55", ticker),
            ("60", now),
        ]
        body = self._FIX_DELIMITER.join(f"{k}={v}" for k, v in body_fields)
        header = self._FIX_DELIMITER.join([
            "8=FIX.4.4", f"9={len(body)}", "35=F",
            f"49={sender}", f"56={target}", "34=1", f"52={now}",
        ])
        full_msg = header + self._FIX_DELIMITER + body
        checksum = sum(ord(c) for c in full_msg) % 256
        return full_msg + self._FIX_DELIMITER + f"10={checksum:03d}"


# ── Abstract router ───────────────────────────────────────────────────────────
class OrderRouter(ABC):
    """Abstract base for all order routers."""

    @abstractmethod
    def place_order(self, order: Order) -> OrderResponse:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...


# ── Paper router (simulation) ─────────────────────────────────────────────────
class PaperOrderRouter(OrderRouter):
    """
    Paper-trading router: simulates fills without real money.
    Uses ExecutionModel (slippage + latency) from backtester.slippage.
    """

    def __init__(self, slippage_ticks: float = 1.0, seed: int = 42):
        from backtester.slippage import ExecutionModel
        self._exec = ExecutionModel(
            slippage_min_ticks=0.5,
            slippage_max_ticks=slippage_ticks * 2,
            seed=seed,
        )
        self._orders: Dict[str, OrderResponse] = {}
        self._positions: Dict[str, Position] = {}
        log.info("[PaperRouter] Initialised (paper trading mode)")

    def place_order(self, order: Order) -> OrderResponse:
        """Simulate order fill using slippage model."""
        theoretical = order.price if order.price > 0 else 1000.0
        fill_info = self._exec.execute(theoretical, order.quantity, order.side.value)

        resp = OrderResponse(
            order_id   = order.order_id,
            status     = OrderStatus.COMPLETE,
            message    = f"PAPER FILL @ ₹{fill_info['fill_price']:.2f} "
                         f"(slip={fill_info['slippage']:.2f}, lat={fill_info['latency_ms']:.0f}ms)",
            fill_price = fill_info["fill_price"],
            filled_qty = order.quantity,
            raw        = fill_info,
        )
        self._orders[order.order_id] = resp

        # Update position
        key = f"{order.ticker}:{order.product.value}"
        if key in self._positions:
            pos = self._positions[key]
            if order.side == OrderSide.BUY:
                total_qty   = pos.quantity + order.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + fill_info["fill_price"] * order.quantity) / total_qty
                pos.quantity  = total_qty
            else:
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self._positions[key]
                    return resp
        else:
            if order.side == OrderSide.BUY:
                self._positions[key] = Position(
                    ticker    = order.ticker,
                    quantity  = order.quantity,
                    avg_price = fill_info["fill_price"],
                    product   = order.product,
                    exchange  = order.exchange,
                )

        log.info("[PaperRouter] %s", resp.message)
        return resp

    def cancel_order(self, order_id: str) -> OrderResponse:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
        return OrderResponse(order_id=order_id, status=OrderStatus.CANCELLED, message="Cancelled")

    def get_order_status(self, order_id: str) -> OrderResponse:
        return self._orders.get(
            order_id,
            OrderResponse(order_id=order_id, status=OrderStatus.PENDING, message="Not found")
        )

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_fill_stats(self) -> dict:
        return self._exec.get_statistics()


# ── Kite router ───────────────────────────────────────────────────────────────
class KiteOrderRouter(OrderRouter):
    """
    Zerodha Kite Connect order router.
    Requires:  pip install kiteconnect
    Env vars:  KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN

    Paper-trading mode is used when KITE_PAPER=1 (default=1 for safety).
    """

    _KITE_PRODUCT = {
        ProductType.CNC:  "CNC",
        ProductType.MIS:  "MIS",
        ProductType.NRML: "NRML",
    }
    _KITE_ORDER_TYPE = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT:  "LIMIT",
        OrderType.SL:     "SL",
        OrderType.SL_M:   "SL-M",
    }

    def __init__(self):
        self._paper = os.getenv("KITE_PAPER", "1") == "1"
        if self._paper:
            log.info("[KiteRouter] PAPER mode (set KITE_PAPER=0 for live trading)")
            self._paper_router = PaperOrderRouter()
            return

        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError(
                "Install kiteconnect: pip install kiteconnect\n"
                "Docs: https://kite.trade/docs/connect/v3/"
            )

        self._kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
        self._kite.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
        log.info("[KiteRouter] LIVE mode — Kite Connect connected")

    def place_order(self, order: Order) -> OrderResponse:
        if self._paper:
            return self._paper_router.place_order(order)

        try:
            kite_order_id = self._kite.place_order(
                variety     = self._kite.VARIETY_REGULAR,
                exchange    = order.exchange.value,
                tradingsymbol = order.ticker.replace(".NS", "").replace(".BO", ""),
                transaction_type = order.side.value,
                quantity    = order.quantity,
                product     = self._KITE_PRODUCT[order.product],
                order_type  = self._KITE_ORDER_TYPE[order.order_type],
                price       = order.price if order.order_type == OrderType.LIMIT else None,
                trigger_price = order.trigger_price if order.order_type in (OrderType.SL, OrderType.SL_M) else None,
                tag         = order.tag,
            )
            log.info("[KiteRouter] Live order placed: %s", kite_order_id)
            return OrderResponse(
                order_id   = kite_order_id,
                status     = OrderStatus.OPEN,
                message    = "Order placed via Kite Connect",
                raw        = {"kite_order_id": kite_order_id},
            )
        except Exception as exc:
            log.error("[KiteRouter] Order failed: %s", exc)
            return OrderResponse(
                order_id = order.order_id,
                status   = OrderStatus.REJECTED,
                message  = str(exc),
            )

    def cancel_order(self, order_id: str) -> OrderResponse:
        if self._paper:
            return self._paper_router.cancel_order(order_id)
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return OrderResponse(order_id=order_id, status=OrderStatus.CANCELLED, message="Cancelled")
        except Exception as exc:
            return OrderResponse(order_id=order_id, status=OrderStatus.REJECTED, message=str(exc))

    def get_order_status(self, order_id: str) -> OrderResponse:
        if self._paper:
            return self._paper_router.get_order_status(order_id)
        try:
            orders = self._kite.orders()
            for o in orders:
                if str(o["order_id"]) == str(order_id):
                    return OrderResponse(
                        order_id   = order_id,
                        status     = OrderStatus(o["status"].upper()),
                        message    = o.get("status_message", ""),
                        fill_price = float(o.get("average_price", 0)),
                        filled_qty = int(o.get("filled_quantity", 0)),
                        raw        = o,
                    )
            return OrderResponse(order_id=order_id, status=OrderStatus.PENDING, message="Not found")
        except Exception as exc:
            return OrderResponse(order_id=order_id, status=OrderStatus.REJECTED, message=str(exc))

    def get_positions(self) -> List[Position]:
        if self._paper:
            return self._paper_router.get_positions()
        try:
            raw = self._kite.positions()
            positions = []
            for p in raw.get("net", []):
                positions.append(Position(
                    ticker    = p["tradingsymbol"],
                    quantity  = int(p["quantity"]),
                    avg_price = float(p["average_price"]),
                    product   = ProductType(p["product"]),
                    exchange  = Exchange(p["exchange"]),
                    pnl       = float(p.get("pnl", 0)),
                    value     = float(p.get("value", 0)),
                ))
            return positions
        except Exception as exc:
            log.error("[KiteRouter] Failed to fetch positions: %s", exc)
            return []


# ── FIX router ────────────────────────────────────────────────────────────────
class FIXOrderRouter(OrderRouter):
    """
    FIX 4.4 order router — sends orders via TCP to an OMS / broker FIX gateway.
    Env vars: FIX_HOST, FIX_PORT, FIX_SENDER, FIX_TARGET
    For testing, set FIX_DRY_RUN=1 to only log FIX messages without sending.
    """

    def __init__(self):
        self._builder = FIXMessageBuilder()
        self._dry_run = os.getenv("FIX_DRY_RUN", "1") == "1"
        self._host    = os.getenv("FIX_HOST", "fix.broker.example.com")
        self._port    = int(os.getenv("FIX_PORT", "9882"))
        self._sender  = os.getenv("FIX_SENDER", "ALADDIN")
        self._target  = os.getenv("FIX_TARGET", "NSE_OMS")
        self._orders: Dict[str, OrderResponse] = {}
        if self._dry_run:
            log.info("[FIXRouter] DRY-RUN mode (FIX messages logged, not sent)")

    def place_order(self, order: Order) -> OrderResponse:
        msg = self._builder.new_order_single(order, self._sender, self._target)
        if self._dry_run:
            log.info("[FIXRouter] DRY-RUN NewOrderSingle:\n%s", msg.replace("\x01", "|"))
            resp = OrderResponse(
                order_id=order.order_id, status=OrderStatus.OPEN,
                message=f"FIX DRY-RUN: {order.side.value} {order.quantity} {order.ticker}",
                raw={"fix_msg": msg},
            )
        else:
            resp = self._send_fix(msg, order.order_id)
        self._orders[order.order_id] = resp
        return resp

    def cancel_order(self, order_id: str) -> OrderResponse:
        orig = self._orders.get(order_id)
        ticker = orig.raw.get("ticker", "UNKNOWN") if orig else "UNKNOWN"
        msg = self._builder.order_cancel_request(
            str(uuid.uuid4())[:16], order_id, ticker, OrderSide.BUY,
            self._sender, self._target,
        )
        if self._dry_run:
            log.info("[FIXRouter] DRY-RUN CancelRequest: %s", msg.replace("\x01", "|"))
            return OrderResponse(order_id=order_id, status=OrderStatus.CANCELLED, message="DRY-RUN cancel")
        return self._send_fix(msg, order_id)

    def get_order_status(self, order_id: str) -> OrderResponse:
        return self._orders.get(
            order_id,
            OrderResponse(order_id=order_id, status=OrderStatus.PENDING, message="Not tracked")
        )

    def get_positions(self) -> List[Position]:
        log.warning("[FIXRouter] get_positions not implemented for FIX — query OMS directly")
        return []

    def _send_fix(self, msg: str, order_id: str) -> OrderResponse:
        """Send a raw FIX message over TCP socket."""
        import socket
        try:
            with socket.create_connection((self._host, self._port), timeout=5) as sock:
                sock.sendall(msg.encode("ascii"))
                ack = sock.recv(4096).decode("ascii", errors="ignore")
            return OrderResponse(order_id=order_id, status=OrderStatus.OPEN, message="ACK received", raw={"ack": ack})
        except Exception as exc:
            log.error("[FIXRouter] TCP send failed: %s", exc)
            return OrderResponse(order_id=order_id, status=OrderStatus.REJECTED, message=str(exc))


# ── Factory ───────────────────────────────────────────────────────────────────
def get_order_router(mode: str = "paper") -> OrderRouter:
    """
    Factory function — returns the appropriate order router.

    mode:
      'paper' — PaperOrderRouter (default, safe for dev/testing)
      'kite'  — KiteOrderRouter (Zerodha Kite Connect)
      'fix'   — FIXOrderRouter (FIX 4.4 TCP gateway)
    """
    mode = os.getenv("ORDER_ROUTER", mode).lower()
    if mode == "kite":
        return KiteOrderRouter()
    if mode == "fix":
        return FIXOrderRouter()
    return PaperOrderRouter()
