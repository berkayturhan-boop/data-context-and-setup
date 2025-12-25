import pandas as pd
import numpy as np
from olist.data import Olist
from olist.utils import haversine_distance

class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # 1. self.data içinden orders tablosunu alıyoruz
        orders = self.data['orders'].copy()

        # 2. İstenirse sadece 'delivered' (teslim edilmiş) olanları alıyoruz
        if is_delivered:
            orders = orders[orders['order_status'] == 'delivered']

        # 3. Tarih formatlarını düzeltiyoruz
        date_cols = ['order_purchase_timestamp', 
                     'order_delivered_customer_date', 
                     'order_estimated_delivery_date']
        
        for col in date_cols:
            orders[col] = pd.to_datetime(orders[col])

        # 4. Wait Time (Bekleme Süresi)
        orders['wait_time'] = (orders['order_delivered_customer_date'] - 
                               orders['order_purchase_timestamp']) / pd.Timedelta(days=1)

        # 5. Expected Wait Time (Tahmini Bekleme Süresi)
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] - 
                                        orders['order_purchase_timestamp']) / pd.Timedelta(days=1)

        # 6. Delay vs Expected (Gecikme var mı?)
        orders['delay_vs_expected'] = (orders['order_delivered_customer_date'] - 
                                       orders['order_estimated_delivery_date']) / pd.Timedelta(days=1)

        # 7. Erken teslimatları (negatif sayıları) 0 yap
        orders['delay_vs_expected'] = orders['delay_vs_expected'].clip(lower=0)

        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()
        
        # 5 yıldız ve 1 yıldız durumlarını 1/0 olarak kodluyoruz
        reviews['dim_is_five_star'] = (reviews['review_score'] == 5).astype(int)
        reviews['dim_is_one_star'] = (reviews['review_score'] == 1).astype(int)

        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        data = self.data['order_items'].copy()
        items = data.groupby('order_id').count()
        return items[['order_item_id']].rename(columns={'order_item_id': 'number_of_items'})

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        data = self.data['order_items'].copy()
        sellers = data.groupby('order_id')['seller_id'].nunique()
        return sellers.to_frame().rename(columns={'seller_id': 'number_of_sellers'})

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        data = self.data['order_items'].copy()
        return data.groupby('order_id')[['price', 'freight_value']].sum()

    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        # 1. Gerekli tabloları al
        geo = self.data['geolocation'].copy()
        orders = self.data['orders'].copy()
        order_items = self.data['order_items'].copy()
        customers = self.data['customers'].copy()
        sellers = self.data['sellers'].copy()

        # 2. Geolocation verisini grupla (ortalama koordinatlar)
        geo = geo.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        })

        # 3. Tabloları birleştir (Merge)
        items_sellers = order_items.merge(sellers, on='seller_id')
        orders_customers = orders.merge(customers, on='customer_id')

        items_sellers = items_sellers[['order_id', 'seller_id', 'seller_zip_code_prefix']]
        orders_customers = orders_customers[['order_id', 'customer_id', 'customer_zip_code_prefix']]
        
        data = items_sellers.merge(orders_customers, on='order_id')

        # 4. Koordinatları ekle
        data = data.merge(geo, left_on='seller_zip_code_prefix', right_index=True)
        data = data.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})

        data = data.merge(geo, left_on='customer_zip_code_prefix', right_index=True)
        data = data.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})

        # 5. Mesafeyi hesapla
        data['distance_seller_customer'] = data.apply(
            lambda row: haversine_distance(
                row['seller_lng'], 
                row['seller_lat'], 
                row['customer_lng'], 
                row['customer_lat']
            ), axis=1
        )
        
        return data.groupby('order_id', as_index=False)[['distance_seller_customer']].mean()

    def get_training_data(self, is_delivered=True, with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # 1. Ana iskeleti oluştur: Wait Time
        training_data = self.get_wait_time(is_delivered=is_delivered)
        
        # 2. Diğer özellikleri ekle
        training_data = training_data.merge(self.get_review_score(), on='order_id')
        training_data = training_data.merge(self.get_number_items(), on='order_id')
        training_data = training_data.merge(self.get_number_sellers(), on='order_id')
        training_data = training_data.merge(self.get_price_and_freight(), on='order_id')
        
        # 3. Mesafe hesabı istenirse ekle
        if with_distance_seller_customer:
            training_data = training_data.merge(self.get_distance_seller_customer(), on='order_id')
        
        # 4. Eksik verileri temizle ve döndür
        return training_data.dropna()