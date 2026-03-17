//+------------------------------------------------------------------+
//|                                              SignalBridge.mq5     |
//|  Reads latest_signal.json from Python framework and executes      |
//|  multi-instrument trades on MT5.                                  |
//|                                                                   |
//|  JSON format expected:                                            |
//|  {                                                                |
//|    "timestamp": "...",                                            |
//|    "count": N,                                                    |
//|    "action": "open" | "none",                                     |
//|    "signals": [ { signal objects }, ... ]                         |
//|  }                                                                |
//+------------------------------------------------------------------+
#property copyright "Trading Framework"
#property version   "2.00"
#property strict

#include "JsonParser.mqh"

//--- Input parameters
input string   InpSignalPath     = "latest_signal.json";  // In MQL5\Files\ folder
input int      InpPollSeconds    = 5;          // How often to check for new signals
input int      InpMagicNumber    = 99999;      // EA magic number
input double   InpMaxSlippage    = 3.0;        // Max slippage in points
input bool     InpCloseOnOpposite = true;      // Close opposite position before opening
input int      InpMaxOpenTrades  = 3;          // Max simultaneous open trades
input int      InpSignalExpiry   = 60;         // Ignore signals older than N minutes

//--- Global state
datetime g_lastSignalTime = 0;                 // Timestamp of last processed signal
string   g_lastOrderIds[];                     // Order IDs already processed
datetime g_lastPollTime   = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("SignalBridge EA initialized");
   Print("Signal file: ", InpSignalPath);
   Print("Poll interval: ", InpPollSeconds, "s");
   Print("Magic number: ", InpMagicNumber);

   EventSetTimer(InpPollSeconds);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("SignalBridge EA removed. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer event — poll the signal file                                |
//+------------------------------------------------------------------+
void OnTimer()
{
   ProcessSignalFile();
}

//+------------------------------------------------------------------+
//| Tick event — also poll (fallback if timer fails)                  |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime now = TimeCurrent();
   if(now - g_lastPollTime < InpPollSeconds)
      return;
   g_lastPollTime = now;
   ProcessSignalFile();
}

//+------------------------------------------------------------------+
//| Main signal processing logic                                      |
//+------------------------------------------------------------------+
void ProcessSignalFile()
{
   //--- Read the JSON file
   string json = ReadFileContents(InpSignalPath);
   if(json == "")
      return;

   //--- Parse the batch wrapper
   string action = JsonGetString(json, "action");
   if(action == "none" || action == "")
      return;

   int count = (int)JsonGetDouble(json, "count");
   if(count <= 0)
      return;

   string timestamp = JsonGetString(json, "timestamp");

   //--- Check signal freshness
   datetime signalTime = ParseISO8601(timestamp);
   if(signalTime <= 0)
   {
      Print("WARNING: Could not parse signal timestamp: ", timestamp);
      return;
   }

   // Skip if we already processed this timestamp
   if(signalTime <= g_lastSignalTime)
      return;

   // Skip if signal is too old
   datetime now = TimeGMT();
   int ageMinutes = (int)((now - signalTime) / 60);
   if(ageMinutes > InpSignalExpiry)
   {
      Print("Signal expired: age=", ageMinutes, " min > ", InpSignalExpiry, " min");
      g_lastSignalTime = signalTime;
      return;
   }

   //--- Parse each signal in the array
   string signals[];
   int nSignals = JsonGetArray(json, "signals", signals);

   Print("Processing ", nSignals, " signal(s) from ", timestamp);

   int executed = 0;
   for(int i = 0; i < nSignals; i++)
   {
      string sig = signals[i];

      string orderId     = JsonGetString(sig, "order_id");
      string asset       = JsonGetString(sig, "asset");
      string side        = JsonGetString(sig, "side");
      string strategy    = JsonGetString(sig, "strategy");
      string regime      = JsonGetString(sig, "regime");
      double entry       = JsonGetDouble(sig, "entry");
      double stopLoss    = JsonGetDouble(sig, "stop_loss");
      double takeProfit  = JsonGetDouble(sig, "take_profit");
      double lots        = JsonGetDouble(sig, "lots");
      double riskPct     = JsonGetDouble(sig, "risk_pct");
      double strength    = JsonGetDouble(sig, "signal_strength");
      int    magic       = (int)JsonGetDouble(sig, "magic_number");
      string comment     = JsonGetString(sig, "comment");

      if(magic == 0) magic = InpMagicNumber;

      //--- Skip if already processed
      if(IsOrderProcessed(orderId))
      {
         Print("Skipping already processed order: ", orderId);
         continue;
      }

      //--- Validate
      if(asset == "" || lots <= 0 || stopLoss <= 0 || takeProfit <= 0)
      {
         Print("Invalid signal values for ", asset, ": lots=", lots,
               " sl=", stopLoss, " tp=", takeProfit);
         continue;
      }

      //--- Map asset name to MT5 symbol
      string symbol = MapToMT5Symbol(asset);
      if(symbol == "")
      {
         Print("Unknown asset mapping for: ", asset);
         continue;
      }

      //--- Check max open trades
      if(CountOpenTrades(magic) >= InpMaxOpenTrades)
      {
         Print("Max open trades (", InpMaxOpenTrades, ") reached. Skipping ", symbol);
         continue;
      }

      //--- Close opposite position if configured
      if(InpCloseOnOpposite)
         CloseOppositePosition(symbol, side, magic);

      //--- Execute the trade
      bool success = ExecuteTrade(symbol, side, lots, stopLoss, takeProfit,
                                   magic, comment, orderId);
      if(success)
      {
         MarkOrderProcessed(orderId);
         executed++;
         Print("EXECUTED: ", symbol, " ", side, " ", lots, " lots | ",
               "SL=", stopLoss, " TP=", takeProfit, " | strategy=", strategy,
               " regime=", regime, " strength=", strength);
      }
   }

   g_lastSignalTime = signalTime;
   Print("Batch complete: ", executed, "/", nSignals, " signal(s) executed");
}

//+------------------------------------------------------------------+
//| Read entire file into a string                                    |
//+------------------------------------------------------------------+
string ReadFileContents(string path)
{
   // Try MQL5\Files\ sandbox first (FILE_SHARE_READ so Python can write simultaneously)
   int handle = FileOpen(path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);

   // Try common files folder as fallback
   if(handle == INVALID_HANDLE)
      handle = FileOpen(path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);

   if(handle == INVALID_HANDLE)
      return "";

   string content = "";
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle);
      if(!FileIsEnding(handle))
         content += "\n";
   }
   FileClose(handle);
   return content;
}

//+------------------------------------------------------------------+
//| Map Python asset names to MT5 symbol names                        |
//| Adjust these mappings to match your broker's symbol naming!       |
//+------------------------------------------------------------------+
string MapToMT5Symbol(string asset)
{
   // Common mappings — EDIT THESE for your broker
   if(asset == "EURUSD") return "EURUSD";
   if(asset == "GBPUSD") return "GBPUSD";
   if(asset == "USDJPY") return "USDJPY";
   if(asset == "AUDUSD") return "AUDUSD";
   if(asset == "XAUUSD") return "XAUUSD";
   if(asset == "BTCUSD") return "BTCUSD";
   if(asset == "Nasdaq") return "Nasdaq";  // Broker-specific (also try NAS100, USTEC, US100)

   // Try the asset name directly (some brokers use standard names)
   if(SymbolInfoInteger(asset, SYMBOL_EXIST))
      return asset;

   // Try with suffix (e.g. EURUSD.i, EURUSDm)
   string suffixes[] = {".i", "m", ".raw", ".pro", ".ecn"};
   for(int i = 0; i < ArraySize(suffixes); i++)
   {
      string test = asset + suffixes[i];
      if(SymbolInfoInteger(test, SYMBOL_EXIST))
         return test;
   }

   return "";
}

//+------------------------------------------------------------------+
//| Execute a market order                                            |
//+------------------------------------------------------------------+
bool ExecuteTrade(string symbol, string side, double lots, double sl,
                  double tp, int magic, string comment, string orderId)
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = NormalizeLots(symbol, lots);
   request.deviation = (ulong)InpMaxSlippage;
   request.magic     = magic;
   request.comment   = comment + "|" + orderId;
   request.type_filling = GetFillingMode(symbol);

   if(side == "long")
   {
      request.type = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      request.sl = NormalizePrice(symbol, sl);
      request.tp = NormalizePrice(symbol, tp);
   }
   else if(side == "short")
   {
      request.type = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
      request.sl = NormalizePrice(symbol, sl);
      request.tp = NormalizePrice(symbol, tp);
   }
   else
   {
      Print("Unknown side: ", side);
      return false;
   }

   //--- Validate before sending
   if(request.volume <= 0)
   {
      Print("Invalid lot size for ", symbol, ": ", lots, " -> ", request.volume);
      return false;
   }

   if(request.price <= 0)
   {
      Print("No price available for ", symbol);
      return false;
   }

   //--- Send order
   ResetLastError();
   bool ok = OrderSend(request, result);

   if(!ok || result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderSend FAILED for ", symbol, ": retcode=", result.retcode,
            " error=", GetLastError(), " comment=", result.comment);
      return false;
   }

   Print("Order filled: ticket=", result.deal, " ", symbol, " ", side,
         " ", request.volume, " lots @ ", result.price);
   return true;
}

//+------------------------------------------------------------------+
//| Close any existing position in the opposite direction             |
//+------------------------------------------------------------------+
void CloseOppositePosition(string symbol, string newSide, int magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic)  continue;

      long posType = PositionGetInteger(POSITION_TYPE);
      bool isOpposite = false;

      if(newSide == "long"  && posType == POSITION_TYPE_SELL) isOpposite = true;
      if(newSide == "short" && posType == POSITION_TYPE_BUY)  isOpposite = true;

      if(isOpposite)
      {
         Print("Closing opposite position: ticket=", ticket, " ", symbol);
         ClosePosition(ticket, symbol, posType);
      }
   }
}

//+------------------------------------------------------------------+
//| Close a specific position                                         |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, string symbol, long posType)
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};

   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = symbol;
   request.volume   = PositionGetDouble(POSITION_VOLUME);
   request.position = ticket;
   request.deviation = (ulong)InpMaxSlippage;
   request.type_filling = GetFillingMode(symbol);

   if(posType == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }

   bool ok = OrderSend(request, result);
   if(!ok || result.retcode != TRADE_RETCODE_DONE)
   {
      Print("Close FAILED: ticket=", ticket, " retcode=", result.retcode);
      return false;
   }

   Print("Position closed: ticket=", ticket, " @ ", result.price);
   return true;
}

//+------------------------------------------------------------------+
//| Count open trades for this EA                                     |
//+------------------------------------------------------------------+
int CountOpenTrades(int magic)
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetInteger(POSITION_MAGIC) == magic)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Normalize lot size to broker constraints                          |
//+------------------------------------------------------------------+
double NormalizeLots(string symbol, double lots)
{
   double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(lotStep <= 0) lotStep = 0.01;
   if(minLot  <= 0) minLot  = 0.01;

   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   lots = MathFloor(lots / lotStep) * lotStep;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Normalize price to symbol's digit precision                       |
//+------------------------------------------------------------------+
double NormalizePrice(string symbol, double price)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   return NormalizeDouble(price, digits);
}

//+------------------------------------------------------------------+
//| Get the correct filling mode for the symbol                       |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol)
{
   long fillPolicy = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);

   if((fillPolicy & SYMBOL_FILLING_FOK) != 0)
      return ORDER_FILLING_FOK;
   if((fillPolicy & SYMBOL_FILLING_IOC) != 0)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Parse ISO 8601 timestamp to datetime                              |
//+------------------------------------------------------------------+
datetime ParseISO8601(string iso)
{
   // Format: "2026-03-14T19:42:00.123456+00:00"
   // Extract: YYYY.MM.DD HH:MM:SS
   if(StringLen(iso) < 19) return 0;

   string date_part = StringSubstr(iso, 0, 10);   // 2026-03-14
   string time_part = StringSubstr(iso, 11, 8);    // 19:42:00

   StringReplace(date_part, "-", ".");

   string dt_str = date_part + " " + time_part;
   return StringToTime(dt_str);
}

//+------------------------------------------------------------------+
//| Track processed order IDs to avoid duplicates                     |
//+------------------------------------------------------------------+
bool IsOrderProcessed(string orderId)
{
   for(int i = 0; i < ArraySize(g_lastOrderIds); i++)
   {
      if(g_lastOrderIds[i] == orderId)
         return true;
   }
   return false;
}

void MarkOrderProcessed(string orderId)
{
   int size = ArraySize(g_lastOrderIds);
   // Keep last 50 order IDs to prevent memory growth
   if(size >= 50)
   {
      for(int i = 0; i < size - 1; i++)
         g_lastOrderIds[i] = g_lastOrderIds[i + 1];
      g_lastOrderIds[size - 1] = orderId;
   }
   else
   {
      ArrayResize(g_lastOrderIds, size + 1);
      g_lastOrderIds[size] = orderId;
   }
}
//+------------------------------------------------------------------+
