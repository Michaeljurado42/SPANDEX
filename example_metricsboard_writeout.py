import yaml

if __name__ == '__main__':

    entries = [
        {
            'team': 'SPANDEX',
            'model': '50% Sparsity SDNN',
            'date': '2023-08-18',
            'SI-SNR': 12.33,
            'SI-SNRi_data': 4.75,
            'SI-SNRi_enc+dec': 4.75,
            'MOS_ovrl': 2.70,
            'MOS_sig': 3.19,
            'MOS_bak': 3.46,
            'latency_enc+dec_ms': 0.006,
            'latency_total_ms': 32.006,
            'power_proxy_Ops/s': 9.373 * 10**6,
            'PDP_proxy_Ops': 0.300 * 10**6,
            'params': 344 * 10**3,
            'size_kilobytes': 305,
            'model_path': './baseline_solution/sdnn_delays/prune50/network.pt',
        },
        {
            'team': 'SPANDEX',
            'model': '75% Sparsity SDNN',
            'date': '2023-08-18',
            'SI-SNR': 11.90,
            'SI-SNRi_data': 4.32,
            'SI-SNRi_enc+dec': 4.32,
            'MOS_ovrl': 2.69,
            'MOS_sig': 3.25,
            'MOS_bak': 3.30,
            'latency_enc+dec_ms': 0.006,
            'latency_total_ms': 32.006,
            'power_proxy_Ops/s': 6.04 * 10**6, 
            'PDP_proxy_Ops': .193 *10**6,
            'params': 174 * 10**3,
            'size_kilobytes': 154,
            'model_path': './baseline_solution/sdnn_delays/prune75/network.pt',
        },
        ]
      
    with open('./metricsboard_track_1_validation.yml', 'w') as outfile:
        yaml.dump(entries, outfile)
      
    entries = [
        {
            'team': 'SPANDEX',
            'model': '50% Sparsity SDNN',
            'date': '2023-08-18',
            'SI-SNR': 12.16,
            'SI-SNRi_data': 4.8,
            'SI-SNRi_enc+dec': 4.8,
            'MOS_ovrl': 2.70,
            'MOS_sig': 3.19,
            'MOS_bak': 3.46,
            'latency_enc+dec_ms': 0.006,
            'latency_total_ms': 32.006,
            'power_proxy_Ops/s': 9.323 * 10**6,
            'PDP_proxy_Ops': 0.298 * 10**6,
            'params': 344 * 10**3,
            'size_kilobytes': 305,
            'model_path': './baseline_solution/sdnn_delays/prune50/network.pt',
        },
        {
            'team': 'SPANDEX',
            'model': '75% Sparsity SDNN',
            'date': '2023-08-18',
            'SI-SNR': 11.72,
            'SI-SNRi_data': 4.36,
            'SI-SNRi_enc+dec': 4.36,
            'MOS_ovrl': 2.68,
            'MOS_sig': 3.24,
            'MOS_bak': 3.29,
            'latency_enc+dec_ms': 0.006,
            'latency_total_ms': 32.006,
            'power_proxy_Ops/s': 6.06 * 10**6, 
            'PDP_proxy_Ops': .194 *10**6,
            'params': 174 * 10**3,
            'size_kilobytes': 154,
            'model_path': './baseline_solution/sdnn_delays/prune75/network.pt',
        },
        ]   
    with open('./metricsboard_track_1_test.yml', 'w') as outfile:
        yaml.dump(entries, outfile)