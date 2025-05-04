import okx.Trade as Trade

class TradingAPI:
    def __init__(self, api_key, secret_key, passphrase, flag="1"):
        """
        初始化交易API
        :param api_key: API Key
        :param secret_key: Secret Key
        :param passphrase: Passphrase
        :param flag: '0'表示实盘交易，'1'表示模拟盘交易
        """
        self.tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

    def place_order(self, instId="BTC-USDT", tdMode="cash", side="buy", ordType="market", sz="10000"):
        """
        执行买卖订单
        :param instId: 交易对，默认为 "BTC-USDT"
        :param tdMode: 交易模式，默认为 "cash"
        :param side: 'buy' 或 'sell'
        :param ordType: 订单类型，默认为 "market"
        :param sz: 交易数量，默认为 10000
        :return: 交易结果
        """
        result = self.tradeAPI.place_order(
            instId=instId,
            tdMode=tdMode,
            side=side,
            ordType=ordType,
            sz=sz
        )
        return result

# 使用示例：
if __name__ == "__main__":
    # 初始化 TradingAPI 实例
    api_key = "70df264e-0681-4ff8-9c6b-91476ed65912"
    secret_key = "67774F2EEDBC970B9D5905CEB8C45F41"
    passphrase = "Miaozhipeng1!"
    trading_api = TradingAPI(api_key, secret_key, passphrase, flag="1")

