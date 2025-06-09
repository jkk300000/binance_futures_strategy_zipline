import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import shutil
import exchange_calendars as ec
from exchange_calendars import get_calendar, ExchangeCalendar
from exchange_calendars.always_open import AlwaysOpenCalendar
from zipline import run_algorithm
from zipline.api import order_target_percent, set_max_leverage, record
from zipline.data import bundles
from zipline.utils.run_algo import load_extensions
from zipline.data.bundles import register, ingest
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.api import symbol 
from zipline.utils.calendar_utils import get_calendar

from zipline.data.bundles import bundles
from zipline.data.bundles.csvdir import csvdir_equities
from glob import glob
import pytz
import talib as ta
import warnings
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 바이낸스 API 설정
api_key = ""  # 바이낸스 API 키
api_secret = ""  # 바이낸스 API 시크릿 키
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# 거래 설정
trading_symbol  = 'BTC/USDT'
timeframe = '1d'  # 1시간봉 (리샘플링하여 일봉으로 변환)
leverage = 3
limit = 1000
risk_per_trade = 0.02  # 계좌 잔고의 2% 리스크
stop_loss_pct = 0.02  # 2% 손절
take_profit_pct = 0.04  # 4% 익절
lookback_period = 1  # 학습 데이터 기간
mc_simulations = 1000  # 몬테카를로 시뮬레이션 횟수
initial_capital = 100000  # 백테스팅 초기 자본 (USDT)

# 글로벌 24/7 캘린더 객체
TRADING_CALENDAR = get_calendar('24/7')


# 레버리지 및 격리 마진 설정 (CCXT 방식)
def set_leverage_and_margin(trading_symbol, leverage):
    try:
        binance.set_leverage(leverage, trading_symbol.replace('/', ''))  # CCXT set_leverage 메서드
        binance.set_margin_mode('isolated', trading_symbol.replace('/', ''))  # 격리 마진 설정
        logger.info(f"레버리지 {leverage}배, 격리 마진 설정 완료")
    except Exception as e:
        logger.error(f"레버리지/마진 설정 오류: {e}")

# 차트 데이터 가져오기
def fetch_ohlcv(trading_symbol, timeframe, limit):
    ohlcv = binance.fetch_ohlcv(trading_symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# 기술적 지표 계산 (pandas_ta 사용)
def calculate_indicators(df):
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    
    df['rsi'] = ta.RSI(close, timeperiod=14)
    df['ma20'] = ta.SMA(close, timeperiod=20)
    df['ma50'] = ta.SMA(close, timeperiod= 50)
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod = 26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['atr'] = ta.ATR(high, df['low'], low, timeperiod=14)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365)
    df.dropna(inplace=True)
    return df

# 학습 데이터 준비
def prepare_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 다음 캔들 상승 여부
    features = ['rsi', 'ma20', 'ma50', 'macd', 'macd_signal', 'atr', 'returns', 'volatility']
    X = df[features]
    # print('X 값', X )
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("훈련 데이터가 비어 있습니다. 데이터 전처리 또는 지표 계산을 확인하세요.")
    return X_train, X_test, y_train, y_test, features

# 모델 학습
def train_models(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    return rf_model, xgb_model

# 몬테카를로 시뮬레이션
def monte_carlo_simulation(returns, num_simulations, periods=24):
    simulations = np.zeros((num_simulations, periods))
    mean_return = returns.mean()
    std_return = returns.std()
    for i in range(num_simulations):
        simulations[i] = np.random.normal(mean_return, std_return, periods).cumsum()
    loss_prob = np.mean(simulations[:, -1] < -stop_loss_pct)
    return loss_prob

# 포지션 크기 계산
def calculate_position_size(balance, current_price, atr, risk_per_trade=0.02):
    risk_amount = balance * risk_per_trade
    stop_loss_pips = atr * 2  # ATR 기반 손절 거리
    position_size = risk_amount / (stop_loss_pips * current_price)
    return position_size

# 거래 실행 (실제 거래용)
def execute_trade(trading_symbol, side, quantity, price, stop_price, take_profit_price):
    try:
        order = binance.create_order(
            symbol=trading_symbol,
            type='market',
            side=side,
            amount=quantity
        )
        logger.info(f"{side.upper()} 주문 실행: {quantity} {trading_symbol} @ {price}")
        
        # 손절/익절 주문 설정
        stop_order = binance.create_order(
            symbol=trading_symbol,
            type='stop_market',
            side='sell' if side == 'buy' else 'buy',
            amount=quantity,
            params={'stopPrice': stop_price}
        )
        take_profit_order = binance.create_order(
            symbol=trading_symbol,
            type='limit',
            side='sell' if side == 'buy' else 'buy',
            amount=quantity,
            price=take_profit_price
        )
        logger.info(f"손절 주문: {stop_price}, 익절 주문: {take_profit_price}")
        return order
    except Exception as e:
        logger.error(f"주문 실행 오류: {e}")
        return None

# Zipline용 데이터 준비
def resample_and_save_csv():
    try:
        # Define paths
        csv_root_dir = Path('~/.zipline/data/binance_futures_csv').expanduser()
        csv_dir = csv_root_dir / 'daily'
        csv_path = csv_dir / 'BTCUSDT.csv'
        
        bundle_dir = Path('~/.zipline/data/binance_futures').expanduser()

        # Clear old directory
        if csv_root_dir.exists():
            shutil.rmtree(csv_root_dir)
            logger.info(f"Cleared directory: {csv_root_dir}")
        
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
            logger.info(f"Cleared bundle directory: {bundle_dir}")
        
        
        
        # Create directory
        csv_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/verified: {csv_dir}")

        # Fetch OHLCV data
        df = fetch_ohlcv(trading_symbol, timeframe, limit)
        if df.empty:
            raise ValueError("No data fetched from Binance. Check API keys or network connection.")

        # Resample to daily
        df_zipline = df[['open', 'high', 'low', 'close', 'volume']].resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Normalize index
        df_zipline.index = df_zipline.index.tz_localize(None).normalize()
        df_zipline.index.name = 'date' 
        start_session = pd.Timestamp(df_zipline.index.min(), tz=pytz.UTC)
        end_session = pd.Timestamp(df_zipline.index.max(), tz=pytz.UTC)

        # Save CSV
        df_zipline.to_csv(csv_path, date_format='%Y-%m-%d')
        logger.info(f"CSV file saved: {csv_path}")

        if not csv_path.exists():
            raise FileNotFoundError(f"Failed to create CSV file: {csv_path}")

        # logger.info(f"CSV file contents (first 5 rows):\n{pd.read_csv(csv_path).head()}")

        return df_zipline, str(csv_path), start_session, end_session

    except Exception as e:
        logger.error(f"Error in resample_and_save_csv: {e}")
        raise


# # 커스텀 번들 정의

# def binance_futures_bundle(environ, asset_db_writer, minute_bar_writer, daily_bar_writer, adjustment_writer, calendar, start_session, end_session, cache, show_progress, output_dir=None):
    
#     # Ensure naive timestamps
#     start_session = start_session.tz_localize()
#     end_session = end_session.tz_localize(None)
    
#     # Log calendar used
#     calendar = TRADING_CALENDAR
#     logger.info(f"Bundle calendar: {calendar} (type: {type(calendar)}, name: {calendar.name})")
    
    
    
    
#     # 1. CSV 파일 경로 설정 및 로드
#     # data_dir = environ.get('ZIPLINE_DATA_DIR', os.path.expanduser('~/.zipline/data'))
#     # data_dir = os.path.expanduser('~/.zipline/data')
    
#     data_dir = Path('~/.zipline/data').expanduser()
#     # bundle_dir = os.path.join(data_dir, 'binance_futures')
#     bundle_dir  = Path(data_dir, 'binance_futures')
    
#     # daily_dir = os.path.join(data_dir, 'binance_futures_csv')
#     daily_dir = Path(data_dir, 'binance_futures_csv')
    
    
#     bundle_timestamp = datetime.now().strftime('%Y-%m-%dT%H_%M_%S')
#     # csv_path = os.path.join(daily_dir, 'BTCUSDT.csv')
#     csv_path = Path(daily_dir, 'BTCUSDT.csv')
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
#     df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    
#     # print('df', df)

#     # [df_zipline, csv_path, start_session, end_session] = resample_and_save_csv()

# # ★ 인덱스를 명확히 DatetimeIndex로 변환 (슬래시/하이픈 모두 대응)
#     if not isinstance(df.index, pd.DatetimeIndex):
#         logger.info("Index is not DatetimeIndex. Converting...")
#         df.index = pd.to_datetime(df.index, errors='coerce')

#     # df_zipline.index = df_zipline.index.tz_localize(None).normalize()
#     df.index = df.index.tz_localize(None).normalize()
#     # 2. 데이터 검증 및 정렬 (NaN, 음수, 타입 등)
#     # ... (생략, 위 코드 참고)

#     # 3. 캘린더 세션에 맞춰 인덱스 재정렬 및 UTC 타임존 부여
#     session_dates = calendar.sessions_in_range(start_session, end_session)
#     expected_index = pd.DatetimeIndex(session_dates, tz=None).normalize()
#     df = df.reindex(expected_index, method='ffill')
#     df.index = df.index.tz_localize('UTC')

#     # Write bcolz data
#     bcolz_dir = os.path.join(bundle_dir, bundle_timestamp, 'daily_equities.bcolz')
#     # os.makedirs(bcolz_dir, exist_ok=True)
#     # daily_bar_writer = BcolzDailyBarWriter(
#     #     bcolz_dir,
#     #     calendar,
#     #     start_session,
#     #     end_session
#     # )
    
#     data = [(0, df[['open', 'high', 'low', 'close', 'volume']])]
#     daily_bar_writer.write(data)

    
#     if minute_bar_writer:
#         logger.info("Skipping minute_bar_writer to avoid minute_equities.bcolz")
        
#     # 5. SQLite(.sqlite) 자산 메타데이터 파일 생성
#     equities_df = pd.DataFrame({
#         'sid': [0],
#         'symbol': ['BTCUSDT'],
#         'asset_name': ['BTCUSDT'],
#         'exchange': ['binance'],
#         'start_date': [df.index.min().tz_convert('UTC')],
#         'end_date': [df.index.max().tz_convert('UTC')],
#         'first_traded': [df.index.min().tz_convert('UTC')]
#     })
    
    
#     asset_db_writer.write(equities=equities_df)
        
   
    
#     # 6. 조정 데이터 없음 (필수 호출)
#     adjustment_writer.write()
    
    
# 번들 등록 및 ingest
def register_and_ingest_bundle(start_session, end_session):
    try:
        csv_root_dir = Path('~/.zipline/data/binance_futures_csv').expanduser()
        csv_path = csv_root_dir / 'daily' / 'BTCUSDT.csv'

        if not csv_root_dir.exists():
            raise FileNotFoundError(f"CSV directory does not exist: {csv_root_dir}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
        logger.info(f"Verified directory: {csv_root_dir}")
        logger.info(f"Verified CSV file: {csv_path}")

        if 'binance_futures' in bundles:
            logger.info("Unregistering existing binance_futures bundle")
            from zipline.data.bundles import unregister
            unregister('binance_futures')

        # Convert to timezone-naive for register
        start_session_naive = start_session.tz_localize(None) if start_session.tzinfo else pd.Timestamp(start_session)
        end_session_naive = end_session.tz_localize(None) if end_session.tzinfo else pd.Timestamp(end_session)

        register(
            'binance_futures',
            csvdir_equities(['daily'], csvdir=str(csv_root_dir)),
            calendar_name='24/7',
            start_session=start_session_naive,
            end_session=end_session_naive
        )
        logger.info("Bundle registered: binance_futures")

        bundle_timestamp = pd.Timestamp.now(tz=pytz.UTC).strftime('%Y-%m-%dT%H:%M:%S.%f')
        ingest('binance_futures', show_progress=False)
        logger.info(f"Zipline 데이터 번들 인제스트 완료: {bundle_timestamp}")

    except Exception as e:
        logger.error(f"Zipline 번들 등록/인제스트 오류: {e}")
        raise
    

# Zipline 백테스팅 전략
def initialize(context):
    context.asset = symbol('BTCUSDT')
    context.leverage = leverage
    context.risk_per_trade = risk_per_trade
    context.stop_loss_pct = stop_loss_pct
    context.take_profit_pct = take_profit_pct
    set_max_leverage(context.leverage)
    
    #모델 초기화
    context.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    context.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    context.features = ['rsi', 'ma20', 'ma50', 'macd', 'macd_signal', 'atr', 'volatility']
    context.is_trained = False

def handle_data(context, data):
    # 현재 데이터 가져오기 (일봉 데이터 사용)
    price_history = data.history(context.asset, ['open', 'high', 'low', 'close', 'volume'], bar_count=110, frequency='1d')
    # price_history.index = price_history.index.tz_localize(None).normalize()
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*stack is deprecated.*')
    
    df = calculate_indicators(price_history)
    # print("df 값", df)
    if len(df) < 30:  # 데이터가 충분하지 않으면 handle_data를 건너뜀
        logger.warning("지표 계산 후 데이터가 부족합니다. handle_data를 건너뜁니다.")
    
    # Rest of handle_data logic
    
    # 모델 학습
    if not context.is_trained and len(df) >= 100:
        X_train, _, y_train, _, _ = prepare_data(df)
        context.rf_model.fit(X_train, y_train)
        context.xgb_model.fit(X_train, y_train)
        context.is_trained = True
        logger.info("모델 학습 완료")
    
    # 예측 및 거래
    if context.is_trained:
        latest_data = df[context.features].iloc[-1:]
        rf_pred = context.rf_model.predict(latest_data)[0]
        xgb_pred = context.xgb_model.predict(latest_data)[0]
        loss_prob = monte_carlo_simulation(df['returns'], mc_simulations)
        
        current_price = data.current(context.asset, 'close')
        atr = df['atr'].iloc[-1]
        balance = context.portfolio.cash
        position_size = calculate_position_size(balance, current_price, atr, context.risk_per_trade)
        logger.info(f'rf_pred 값 : {rf_pred}')
        #거래 신호
        if rf_pred == 1 and xgb_pred == 1 and loss_prob < 0.5:
            order_target_percent(context.asset, position_size / current_price)
            record(signal='buy', price=current_price, loss_prob=loss_prob)
            logger.info(f'buy: {current_price}')
        elif rf_pred == 0 and xgb_pred == 0 and loss_prob < 0.3:
            order_target_percent(context.asset, -position_size / current_price)
            record(signal='sell', price=current_price, loss_prob=loss_prob)
            logger.info(f'sell: {current_price}')
        else:
            record(signal='hold', price=current_price, loss_prob=loss_prob)
            logger.info(f'hold: {current_price}')
        
        # if loss_prob < 1:  # Should always trigger for testing
        #     order_target_percent(context.asset, 1.0)  # Use context.asset
        #     record(signal='buy', price=current_price, loss_prob=loss_prob)
        #     logger.info(f"Buy: {current_price}, Loss Prob: {loss_prob}")
        # else:
        #     record(signal='hold', price=current_price, loss_prob=loss_prob)
        #     logger.info(f"Hold: {current_price}, Loss Prob: {loss_prob}")
    
    # 포지션 관리 (손절/익절)
    if context.portfolio.positions:
        position = context.portfolio.positions[context.asset]
        entry_price = position.cost_basis
        if position.amount > 0:  # 롱 포지션
            if current_price <= entry_price * (1 - context.stop_loss_pct):
                order_target_percent(context.asset, 0)
                record(signal='stop_loss', price=current_price)
            elif current_price >= entry_price * (1 + context.take_profit_pct):
                order_target_percent(context.asset, 0)
                record(signal='take_profit', price=current_price)
        elif position.amount < 0:  # 숏 포지션
            if current_price >= entry_price * (1 + context.stop_loss_pct):
                order_target_percent(context.asset, 0)
                record(signal='stop_loss', price=current_price)
            elif current_price <= entry_price * (1 - context.take_profit_pct):
                order_target_percent(context.asset, 0)
                record(signal='take_profit', price=current_price)

# 백테스팅 실행
def run_backtest():
    try:
        [df_zipline, csv_path, start_session, end_session] = resample_and_save_csv()
        register_and_ingest_bundle(start_session, end_session)

        start = pd.Timestamp('2022-12-21', tz=pytz.UTC)
        end = pd.Timestamp('2025-05-23', tz=pytz.UTC)

        bundle_dir = os.path.join(os.path.expanduser('~/.zipline/data'), 'binance_futures')
        # logger.info(f"번들 디렉토리 내용: {glob(os.path.join(bundle_dir, '*'))}")
        # logger.info(f"백테스트 범위: {start} ~ {end}")

        TRADING_CALENDAR = get_calendar('24/7')
        # logger.info(f"TradingCalendar type: {type(TRADING_CALENDAR)}")
        # logger.info(f"TradingCalendar name: {TRADING_CALENDAR.name}")
        # logger.info(f"Is ExchangeCalendar: {isinstance(TRADING_CALENDAR, ExchangeCalendar)}")

        # Convert to timezone-naive for sessions_in_range
        start_session_naive = start_session.tz_localize(None) if start_session.tzinfo else start_session
        end_session_naive = end_session.tz_localize(None) if end_session.tzinfo else end_session
        sessions = TRADING_CALENDAR.sessions_in_range(start_session_naive, end_session_naive)
        # logger.info(f"Trading sessions: {sessions[0]} to {sessions[-1]}, count: {len(sessions)}")

        # logger.info(f"등록된 번들: {list(bundles.keys())}")

        load_extensions(default=True, extensions=[], strict=True, environ=os.environ)
        
        # Convert to timezone-naive for run_algorithm
        start_naive = start.tz_localize(None)
        end_naive = end.tz_localize(None)
        
        result = run_algorithm(
            start=start_naive,
            end=end_naive,
            initialize=initialize,  # Your initialize function
            handle_data=handle_data,  # Your handle_data function
            capital_base=initial_capital,  # Your capital base
            bundle='binance_futures',
            data_frequency="daily",
            trading_calendar=TRADING_CALENDAR,
            environ=os.environ,
            default_extension=True
        )

        returns = result['returns']
        # logger.info(f"returns: {returns}")
        

        # Calculate metrics
        total_return = (result['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        
        if returns.std() == 0 or np.isnan(returns.std()):
            sharpe_ratio = 0
        else:
            sharpe_ratio = np.sqrt(365) * returns.mean() / returns.std()
        
        max_drawdown = (result['portfolio_value'] / result['portfolio_value'].cummax() - 1).min()

        # Log results
        logger.info(f"백테스팅 결과:")
        logger.info(f"총 수익률: {total_return:.2f}%")
        logger.info(f"샤프 비율: {sharpe_ratio:.2f}")
        logger.info(f"최대 손실(MDD): {max_drawdown*100:.2f}%")
        
        # df = fetch_ohlcv(trading_symbol, timeframe, limit)
        # df = calculate_indicators(df)
        # loss_prob = monte_carlo_simulation(df['returns'], mc_simulations)
        # logger.info(f"몬테카를로 손실 확률: {loss_prob}")
        return result

    except Exception as e:
        logger.error(f"백테스팅 실행 오류: {e}")
        raise
# 메인 트레이딩 루프 (실제 거래)
def main():
    set_leverage_and_margin(trading_symbol, leverage)
    
    # 백테스팅 실행
    df = fetch_ohlcv(trading_symbol, timeframe, limit)
    run_backtest()
    
    
    
    # 실제 거래 시작
    # while True:
    #     try:
    #         # 잔고 조회
    #         balance_info = binance.fetch_balance(params={"type": "future"})
    #         balance = balance_info['USDT']['free']
    #         logger.info(f"현재 잔고: {balance} USDT")
            
    #         # 차트 데이터 가져오기
    #         df = fetch_ohlcv(trading_symbol, timeframe, limit)
    #         df = calculate_indicators(df)
            
    #         # 학습 데이터 준비
    #         X_train, X_test, y_train, y_test, features = prepare_data(df)
            
    #         # 모델 학습
    #         rf_model, xgb_model = train_models(X_train, y_train)
            
    #         # 최신 데이터로 예측
    #         latest_data = df[features].iloc[-1:]
    #         rf_pred = rf_model.predict(latest_data)[0]
    #         xgb_pred = xgb_model.predict(latest_data)[0]
            
    #         # 몬테카를로 시뮬레이션
    #         loss_prob = monte_carlo_simulation(df['returns'], mc_simulations)
    #         logger.info(f"몬테카를로 손실 확률: {loss_prob*100:.2f}%")
            
    #         # 거래 신호
    #         current_price = df['close'].iloc[-1]
    #         atr = df['atr'].iloc[-1]
    #         position_size = calculate_position_size(balance, current_price, atr, risk_per_trade)
            
    #         if rf_pred == 1 and xgb_pred == 1 and loss_prob < 0.3:
    #             stop_price = current_price * (1 - stop_loss_pct)
    #             take_profit_price = current_price * (1 + take_profit_pct)
    #             execute_trade(trading_symbol, 'buy', position_size, current_price, stop_price, take_profit_price)
    #         elif rf_pred == 0 and xgb_pred == 0 and loss_prob < 0.3:
    #             stop_price = current_price * (1 + stop_loss_pct)
    #             take_profit_price = current_price * (1 - take_profit_pct)
    #             execute_trade(trading_symbol, 'sell', position_size, current_price, stop_price, take_profit_price)
    #         else:
    #             logger.info("거래 조건 미충족, 대기 중")
            
    #         # 1시간 대기
    #         time.sleep(3600)
            
    #     except Exception as e:
    #         logger.error(f"메인 루프 오류: {e}")
    #         time.sleep(60)

if __name__ == "__main__":
    main()