# Sinh test (Minh, Hoàng)

- format của test được ghi ở docs/instance_format.md
- Các edge case đặc biệt cần sinh được ghi ở docs/edge_cases.md

# Triển khai thuật toán (Thi, Thắng, Phong)

- Task 1: Cài đặt thuật toán theo bài báo: Multi-UAV Disaster Environment Coverage Planning with
  Limited-Endurance. Sau đó test với dataset đã sinh.
- Task 2: Cài thêm một số thuật toán phổ biến (Thực hiện khi xong task 1).

# Paper tham khảo:

Đặt trong thư mục docs/papers

# Bộ instance RescueNet

Hai bộ chỉ chứa instance đã có chứng nhận mức phủ tối ưu trong **10–50%**:

| Lưới | Val | Test | Tổng |
|---|---:|---:|---:|
| 50×50 | 7.524 | 29.772 | 37.296 |
| 100×100 | 7.533 | 29.781 | 37.314 |

K=6, sáu depot khác nhau, pin riêng; mỗi ảnh được chọn có 3 mức ngân sách × 3 seed (42,43,44).
Giữ split nguồn để tránh rò rỉ val/test. Không sinh edge instance hoặc cấu hình không đạt.
Ảnh chưa được chọn chỉ có chỉ mục nguồn riêng trong `data/rescuenet_unselected/`, không thuộc bộ thực nghiệm.

- [Phương pháp sinh và kiểm chứng](docs/rescuenet_experiment_protocol.md).
- [Config 50×50](configs/rescuenet_instances.json), [config 100×100](configs/rescuenet_instances_100x100.json), [manifest nguồn](configs/rescuenet_split.csv).
- [Input mẫu 50×50](docs/examples/rescuenet_10778_50x50_K6_blow_s42.json), [input mẫu 100×100](docs/examples/rescuenet_10778_100x100_K6_blow_s42.json).

[Google Drive của nhóm](https://drive.google.com/drive/folders/1RWrnqaMsz8lVNN6C5uTmxsERGoTavO6Q?usp=sharing).
Dùng hai ZIP mới, giải nén vào thư mục trống `data/instances/rescuenet_50x50/` và `data/instances/rescuenet_100x100/`;
không chồng lên bản cũ. Giữ manifest, certificates, summary và README cùng JSON.

```powershell
python -m pip install -r requirements-instances.txt
python -m unittest discover -s tests -v
python scripts/validate_instances.py data/instances/rescuenet_50x50 --workers 6
python scripts/validate_instances.py data/instances/rescuenet_100x100 --workers 6
```

Để sinh lại từ mask gốc, chạy generator với `--grid-size 50` hoặc `--grid-size 100`.
Code bàn giao gồm generator, validator, evaluator, coverage_certificate và tests.
Data, outputs, môi trường Python, script chia tập cũ, notebook và PDF không thuộc phần commit mới.
