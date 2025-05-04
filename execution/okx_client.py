import okx.Trade as Trade
import okx.Funding as Funding
import okx.MarketData as MarketData
import okx.Account as Account

api_key = "70df264e-0681-4ff8-9c6b-91476ed65912"
secret_key = "67774F2EEDBC970B9D5905CEB8C45F41"
passphrase = "Miaozhipeng1!"
flag = "1"  # live trading: 0, demo trading: 1

tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

result = tradeAPI.place_order(
    instId="BTC-USDT",
    tdMode="cash",
    side="buy",
    ordType="market",
    sz="10000",
)
print(result)
