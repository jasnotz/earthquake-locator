import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import folium
from folium.plugins import MarkerCluster
import webbrowser
import os
import tempfile
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import math
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

class EarthquakeSimulation:
    def __init__(self, root):
        self.root = root
        self.root.title("지진 진앙 및 진원 시뮬레이션")
        self.root.geometry("1200x800")
        
        self.client = Client("IRIS")
        
        self.event_id = tk.StringVar()
        self.event_info = None
        self.stations = []
        self.p_arrivals = []
        self.s_arrivals = []
        
        self.vp = 6.0
        self.vs = 3.5
        
        self.epicenter = None
        self.hypocenter = None
        
        self.create_ui()
        
    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(main_frame, text="지진 데이터 입력", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        right_frame = ttk.LabelFrame(main_frame, text="시뮬레이션 결과", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(left_frame, text="이벤트 ID:").grid(column=0, row=0, sticky=tk.W, pady=2)
        ttk.Entry(left_frame, textvariable=self.event_id, width=20).grid(column=1, row=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="검색", command=self.fetch_event_data).grid(column=2, row=0, padx=5, pady=2)
        
        ttk.Label(left_frame, text="예시: us7000jl3d, us6000jk0t").grid(column=0, row=1, columnspan=3, sticky=tk.W, pady=2)
        
        event_info_frame = ttk.LabelFrame(left_frame, text="이벤트 정보", padding="5")
        event_info_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.event_info_text = tk.Text(event_info_frame, width=40, height=10, wrap=tk.WORD)
        self.event_info_text.pack(fill=tk.BOTH, expand=True)
        self.event_info_text.config(state=tk.DISABLED)
        
        station_frame = ttk.LabelFrame(left_frame, text="관측소 정보", padding="5")
        station_frame.grid(column=0, row=3, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.station_text = tk.Text(station_frame, width=40, height=15, wrap=tk.WORD)
        self.station_text.pack(fill=tk.BOTH, expand=True)
        self.station_text.config(state=tk.DISABLED)
        
        ttk.Button(left_frame, text="진앙/진원 계산", command=self.calculate_location).grid(column=0, row=4, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.map_frame = ttk.Frame(right_frame)
        self.map_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        result_frame = ttk.LabelFrame(right_frame, text="계산 결과", padding="5")
        result_frame.pack(fill=tk.X, expand=False, pady=5)
        
        self.result_text = tk.Text(result_frame, width=60, height=8, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
    
    def fetch_event_data(self):
        event_id = self.event_id.get().strip()
        if not event_id:
            messagebox.showerror("오류", "이벤트 ID를 입력하세요.")
            return
        
        try:
            usgs_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={event_id}&format=geojson"
            response = requests.get(usgs_url)
            
            if response.status_code != 200:
                messagebox.showerror("오류", f"이벤트 ID {event_id}에 대한 데이터를 찾을 수 없습니다. 상태 코드: {response.status_code}")
                return
            
            event_data = response.json()
            
            coords = event_data['geometry']['coordinates']
            lon, lat, depth_km = coords
            magnitude = event_data['properties']['mag']
            mag_type = event_data['properties']['magType']
            event_time = event_data['properties']['time'] / 1000.0
            
            self.event_info = {
                'id': event_id,
                'time': UTCDateTime(event_time),
                'latitude': lat,
                'longitude': lon,
                'depth': depth_km,
                'magnitude': magnitude,
                'magnitude_type': mag_type
            }
            
            self.update_event_info()
            
            self.fetch_stations()
            
        except Exception as e:
            messagebox.showerror("오류", f"데이터 가져오기 실패: {str(e)}")
    
    def update_event_info(self):
        self.event_info_text.config(state=tk.NORMAL)
        self.event_info_text.delete(1.0, tk.END)
        
        info_text = f"이벤트 ID: {self.event_id.get()}\n"
        info_text += f"발생 시간: {self.event_info['time']}\n"
        info_text += f"위도: {self.event_info['latitude']:.4f}°\n"
        info_text += f"경도: {self.event_info['longitude']:.4f}°\n"
        info_text += f"깊이: {self.event_info['depth']:.2f} km\n"
        info_text += f"규모: {self.event_info['magnitude']:.1f} {self.event_info['magnitude_type']}\n"
        
        self.event_info_text.insert(tk.END, info_text)
        self.event_info_text.config(state=tk.DISABLED)
    
    def fetch_stations(self):
        try:
            event_time = self.event_info['time']
            lat = self.event_info['latitude']
            lon = self.event_info['longitude']
            
            inventory = self.client.get_stations(
                latitude=lat,
                longitude=lon,
                maxradius=5.0,
                starttime=event_time,
                endtime=event_time + 60 * 60,
                level="station"
            )
            
            selected_stations = []
            for network in inventory:
                for station in network:
                    selected_stations.append({
                        'network': network.code,
                        'station': station.code,
                        'latitude': station.latitude,
                        'longitude': station.longitude,
                        'elevation': station.elevation/1000.0,
                    })
                    if len(selected_stations) >= 3:
                        break
                if len(selected_stations) >= 3:
                    break
            
            if len(selected_stations) < 3:
                angles = [0, 120, 240]
                distance = 2.0
                
                for i, angle in enumerate(angles):
                    if i < len(selected_stations):
                        continue
                    
                    angle_rad = math.radians(angle)
                    station_lat = lat + distance * math.sin(angle_rad)
                    station_lon = lon + distance * math.cos(angle_rad)
                    
                    selected_stations.append({
                        'network': 'SIM',
                        'station': f'ST{i+1}',
                        'latitude': station_lat,
                        'longitude': station_lon,
                        'elevation': 0.0,
                    })
            
            self.stations = selected_stations
            
            self.simulate_arrivals()
            
            self.update_station_info()
            
        except Exception as e:
            messagebox.showerror("오류", f"관측소 데이터 가져오기 실패: {str(e)}")
            self.create_simulated_stations()
    
    def create_simulated_stations(self):
        lat = self.event_info['latitude']
        lon = self.event_info['longitude']
        
        angles = [0, 120, 240]
        distance = 2.0
        
        self.stations = []
        for i, angle in enumerate(angles):
            angle_rad = math.radians(angle)
            station_lat = lat + distance * math.sin(angle_rad)
            station_lon = lon + distance * math.cos(angle_rad)
            
            self.stations.append({
                'network': 'SIM',
                'station': f'ST{i+1}',  # 여기서 구문 오류가 수정됨
                'latitude': station_lat,
                'longitude': station_lon,
                'elevation': 0.0,
            })
        
        self.simulate_arrivals()
        
        self.update_station_info()
    
    def simulate_arrivals(self):
        self.p_arrivals = []
        self.s_arrivals = []
        
        event_time = self.event_info['time']
        event_lat = self.event_info['latitude']
        event_lon = self.event_info['longitude']
        event_depth = self.event_info['depth']
        
        for station in self.stations:
            try:
                surface_distance = self.calculate_distance(
                    event_lat, event_lon, 
                    station['latitude'], station['longitude']
                )
                
                distance_3d = math.sqrt(surface_distance**2 + event_depth**2)
                
                p_travel_time = distance_3d / self.vp
                s_travel_time = distance_3d / self.vs
                
                p_arrival = event_time + p_travel_time
                s_arrival = event_time + s_travel_time
                
                error_range = 0.5
                p_error = (np.random.random() - 0.5) * error_range
                s_error = (np.random.random() - 0.5) * error_range
                
                p_arrival += p_error
                s_arrival += s_error
                
                self.p_arrivals.append(p_arrival)
                self.s_arrivals.append(s_arrival)
                
            except Exception as e:
                surface_distance = self.calculate_distance(
                    event_lat, event_lon, 
                    station['latitude'], station['longitude']
                )
                distance_3d = math.sqrt(surface_distance**2 + event_depth**2)
                
                p_travel_time = distance_3d / self.vp
                s_travel_time = distance_3d / self.vs
                
                p_arrival = event_time + p_travel_time
                s_arrival = event_time + s_travel_time
                
                self.p_arrivals.append(p_arrival)
                self.s_arrivals.append(s_arrival)
    
    def update_station_info(self):
        self.station_text.config(state=tk.NORMAL)
        self.station_text.delete(1.0, tk.END)
        
        for i, station in enumerate(self.stations):
            station_text = f"관측소 {i+1}: {station['network']}.{station['station']}\n"
            station_text += f"  위도: {station['latitude']:.4f}°\n"
            station_text += f"  경도: {station['longitude']:.4f}°\n"
            station_text += f"  고도: {station['elevation']:.2f} km\n"
            
            if i < len(self.p_arrivals) and self.p_arrivals[i]:
                station_text += f"  P파 도착 시간: {self.p_arrivals[i]}\n"
            
            if i < len(self.s_arrivals) and self.s_arrivals[i]:
                station_text += f"  S파 도착 시간: {self.s_arrivals[i]}\n"
            
            station_text += "\n"
            
            self.station_text.insert(tk.END, station_text)
        
        self.station_text.config(state=tk.DISABLED)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def objective_function(self, params, stations, p_arrivals, s_arrivals, vp, vs):
        lat, lon, depth, t0 = params
        error_sum = 0
        
        for i, station in enumerate(stations):
            surface_distance = self.calculate_distance(lat, lon, station['latitude'], station['longitude'])
            distance_3d = math.sqrt(surface_distance**2 + depth**2)
            
            calc_p_time = t0 + distance_3d / vp
            calc_s_time = t0 + distance_3d / vs
            
            obs_p_time = p_arrivals[i].timestamp
            obs_s_time = s_arrivals[i].timestamp
            
            p_error = (calc_p_time - obs_p_time) ** 2
            s_error = (calc_s_time - obs_s_time) ** 2
            
            error_sum += p_error + s_error
        
        return error_sum
    
    def calculate_location(self):
        if not self.event_info or not self.stations or len(self.stations) < 3:
            messagebox.showerror("오류", "지진 데이터와 최소 3개의 관측소 데이터가 필요합니다.")
            return
        
        try:
            initial_lat = self.event_info['latitude']
            initial_lon = self.event_info['longitude']
            initial_depth = self.event_info['depth']
            initial_time = self.event_info['time'].timestamp
            
            initial_guess = [initial_lat, initial_lon, initial_depth, initial_time]
            
            result = minimize(
                self.objective_function, 
                initial_guess,
                args=(self.stations, self.p_arrivals, self.s_arrivals, self.vp, self.vs),
                method='Nelder-Mead'
            )
            
            lat, lon, depth, t0 = result.x
            self.epicenter = (lat, lon)
            self.hypocenter = (lat, lon, depth)
            
            self.display_results(lat, lon, depth, t0)
            
            self.display_map(lat, lon, depth)
            
        except Exception as e:
            messagebox.showerror("오류", f"계산 중 오류 발생: {str(e)}")
    
    def display_results(self, lat, lon, depth, t0):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        origin_time = datetime.fromtimestamp(t0)
        
        result_text = f"계산된 진앙:\n"
        result_text += f"  위도: {lat:.4f}°\n"
        result_text += f"  경도: {lon:.4f}°\n"
        result_text += f"진원 깊이: {depth:.2f} km\n"
        result_text += f"발생 시간: {origin_time}\n\n"
        
        result_text += f"카탈로그 데이터 (참고):\n"
        result_text += f"  위도: {self.event_info['latitude']:.4f}°\n"
        result_text += f"  경도: {self.event_info['longitude']:.4f}°\n"
        result_text += f"  깊이: {self.event_info['depth']:.2f} km\n"
        
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
    
    def display_map(self, lat, lon, depth):
        m = folium.Map(
            location=[lat, lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"진앙 (위도: {lat:.4f}°, 경도: {lon:.4f}°)",
            tooltip="진앙 (지표면 위치)"
        ).add_to(m)
        
        radius_scale = 10000
        folium.Circle(
            location=[lat, lon],
            radius=depth * radius_scale,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.4,
            popup=f"진원 (깊이: {depth:.2f} km)",
            tooltip=f"진원 ({depth:.2f}km 지하)"
        ).add_to(m)
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px;
                    border-radius: 5px;">
            <p><b>지진 위치:</b></p>
            <p><i class="fa fa-circle" style="color:red"></i> 진앙 (지표면)</p>
            <p><i class="fa fa-circle" style="color:blue"></i> 진원 ({:.2f}km 깊이)</p>
            <p><small>* 파란색 원의 크기는 깊이에 비례합니다</small></p>
        </div>
        '''.format(depth)
        m.get_root().html.add_child(folium.Element(legend_html))
        
        for i, station in enumerate(self.stations):
            folium.Marker(
                location=[station['latitude'], station['longitude']],
                popup=f"관측소 {i+1}: {station['network']}.{station['station']}",
                tooltip=f"관측소 {i+1}",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
        
        for station in self.stations:
            folium.PolyLine(
                locations=[[lat, lon], [station['latitude'], station['longitude']]],
                color='purple',
                weight=2,
                opacity=0.7,
                tooltip=f"거리: {self.calculate_distance(lat, lon, station['latitude'], station['longitude']):.2f} km"
            ).add_to(m)
        
        for i, station in enumerate(self.stations):
            if i < len(self.p_arrivals) and i < len(self.s_arrivals):
                p_time = self.p_arrivals[i].timestamp
                s_time = self.s_arrivals[i].timestamp
                
                event_time = self.event_info['time'].timestamp
                
                p_travel_time = p_time - event_time
                s_travel_time = s_time - event_time
                
                p_distance = p_travel_time * self.vp * 1000
                s_distance = s_travel_time * self.vs * 1000
                
                folium.Circle(
                    location=[lat, lon],
                    radius=p_distance,
                    color='lightblue',
                    fill=False,
                    weight=1,
                    popup=f"P파 범위 ({p_travel_time:.2f}초)"
                ).add_to(m)
                
                folium.Circle(
                    location=[lat, lon],
                    radius=s_distance,
                    color='pink',
                    fill=False,
                    weight=1,
                    popup=f"S파 범위 ({s_travel_time:.2f}초)"
                ).add_to(m)
        
        temp_html = os.path.join(tempfile.gettempdir(), 'earthquake_map.html')
        m.save(temp_html)
        
        webbrowser.open('file://' + os.path.abspath(temp_html))

if __name__ == "__main__":
    root = tk.Tk()
    app = EarthquakeSimulation(root)
    root.mainloop()