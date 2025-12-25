from pathlib import Path
import pandas as pd


class Olist:
    """
    The Olist class provides methods to interact with Olist's e-commerce data.

    Methods:
        get_data():
            Loads and returns a dictionary where keys are dataset names (e.g., 'sellers', 'orders')
            and values are pandas DataFrames loaded from corresponding CSV files.

        ping():
            Prints "pong" to confirm the method is callable.
    """
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # 1. CSV dosyalarının olduğu yolu belirle (Home klasöründen başlayarak)
        # Path.home() senin bilgisayarındaki ana kullanıcı klasörünü bulur (~ işareti gibi)
        csv_path = Path.home() / ".workintech/olist/data/csv"

        # 2. Dosya isimlerini temizleyip (key), veriyi (value) içine atacağımız boş sözlük
        data = {}

        # 3. Klasördeki tüm .csv dosyalarını gez
        # glob("*.csv") komutu o klasördeki sonu .csv ile biten her şeyi bulur
        for file in csv_path.glob("*.csv"):
            # Dosya ismini al (örn: olist_sellers_dataset) ve temizle -> sellers
            # .stem uzantısız dosya ismini verir, replace ile fazlalıkları atarız
            key = file.stem.replace("olist_", "").replace("_dataset", "").replace("_data", "")

            # Pandas ile oku ve sözlüğe ekle
            data[key] = pd.read_csv(file)

        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
