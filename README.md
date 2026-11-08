# Sinh test (Minh, Hoàng)

- format của test được ghi ở docs/instance_format.md
- Các edge case đặc biệt cần sinh được ghi ở docs/edge_cases.md

# Triển khai thuật toán (Thi, Thắng, Phong)

- Task 1: Cài đặt thuật toán theo bài báo: Multi-UAV Disaster Environment Coverage Planning with
  Limited-Endurance. Sau đó test với dataset đã sinh.
- Task 2: Cài thêm một số thuật toán phổ biến (Thực hiện khi xong task 1).

# Bộ instance RescueNet

Bộ dữ liệu gồm **40.446 instance K=6**, lưới 50×50: **8.091 val** và **32.355 test**.
Mỗi ảnh có 3 mức ngân sách × 3 seed (42, 43, 44); sáu UAV có depot riêng và pin khác nhau.

- [Phương pháp sinh và thực nghiệm](docs/rescuenet_experiment_protocol.md).
- [Cấu hình sinh](configs/rescuenet_instances.json) và [manifest nguồn](configs/rescuenet_split.csv).
- [Instance mẫu](docs/examples/rescuenet_10778_50x50_K6_blow_s42.json).

[Tải bộ instance đầy đủ trên Google Drive](https://drive.google.com/drive/folders/1RWrnqaMsz8lVNN6C5uTmxsERGoTavO6Q?usp=sharing),
giải nén vào `data/instances/rescuenet_50x50/`.
Dùng val chọn tham số, giữ cố định tham số khi đánh giá test.

```powershell
python -m pip install -r requirements-instances.txt
python -m unittest discover -s tests -v
python scripts/validate_instances.py --workers 6
```

Code bàn giao gồm generator, validator, evaluator và tests. Dữ liệu, outputs, môi trường Python,
script chia tập cũ, notebook và PDF không thuộc phần commit mới.
