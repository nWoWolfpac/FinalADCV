SegNet - Backbone compatibility and notes
========================================

Tổng quan
---------
File `src/models/segnet.py` triển khai một decoder kiểu SegNet được thiết kế để hoạt động cùng các encoder có cấu trúc "ResNet-like" (conv1/stem -> layer1 -> layer2 -> layer3 -> layer4). Do đó SegNet trong repo này chỉ tương thích an toàn với các backbone ResNet (ví dụ: `resnet18`, `resnet50`, `resnet101`).

Tại sao SegNet không tương thích với MobileViT / MobileNetV4
-----------------------------------------------------------
- ResNet có cấu trúc phân cấp rõ ràng: một stem (conv1 + bn + act + maxpool) theo sau là 4 block (layer1..layer4). Decoder SegNet dựa trên giả định đó để lấy các feature maps ở 4 mức phân giải và upsample lần lượt.
- MobileViT và MobileNetV4 (và nhiều backbone "mobile" khác) dùng các block nhỏ, depthwise separable convs, inverted residuals, attention blocks, v.v. Chúng không phân cấp thành 4 block rõ ràng cùng tên và thứ tự như ResNet.
- Nếu SegNet cố truy cập các module theo tên hoặc chỉ số (ví dụ: `children()[0]` → `conv1`, `children()[1]` → `layer1`...), mapping này sẽ sai trên backbone mobile. Hậu quả: mismatch về kích thước (H×W) hoặc số channel, gây lỗi runtime hoặc kết quả đào tạo rất kém.

Ví dụ lỗi thường gặp
---------------------
- RuntimeError về channel mismatch (ví dụ: "expected input to have 12 channels, but got 3"), xuất hiện khi code thay đổi/hoán đổi conv đầu hoặc áp adapter không đúng chỗ.
- Giá trị feature không đúng độ phân giải (decoder upsample theo giả định nhưng encoder mobile không sinh feature ở các mức tương ứng).

Giải pháp khuyến nghị (ngắn gọn)
--------------------------------
1. Nếu bạn muốn dùng backbone mobile (MobileViT / MobileNetV4), dùng `DeepLabV3Plus` thay vì `SegNet`.
   - `DeepLabV3Plus` trong repository (`src/models/deeplabv3plus.py`) đã hỗ trợ chia low/high level cho nhiều backbone và xử lý input channels khác nhau.

2. Nếu bạn thật sự cần dùng SegNet với backbone mobile (cần công việc thêm):
   - Cách an toàn hơn: giữ nguyên conv đầu của encoder (không mutate conv1), và thêm một adapter 1x1 ở đầu để chuyển input (ví dụ 12-band) về số channel mà encoder thực tế mong đợi.
   - Cách tổng quát (phức tạp): dùng forward hooks để thu tất cả intermediate activations của encoder trong một dummy forward, chọn những activation đại diện cho nhiều mức phân giải (theo kích thước spatial) và dùng chúng làm c0..c4 cho decoder. Giải pháp này cần test kỹ và có thể phải tinh chỉnh mapping.
   - Cách thủ công: nghiên cứu cấu trúc cụ thể của phiên bản backbone bạn dùng và hard-code mapping (stem/layer1..layer4) tương ứng. Đây là nhanh cho một phiên bản cụ thể nhưng dễ vỡ khi thay phiên bản khác.

Hướng dẫn chạy (ví dụ)
----------------------
- Chạy SegNet với ResNet (được hỗ trợ):

```bash
python training_decoder.py --model segnet --backbone resnet50 --num_classes 8
```

- Chạy DeepLabV3Plus với MobileViT (nếu bạn muốn backbone mobile):

```bash
python training_decoder.py --model deeplabv3 --backbone mobilevit --num_classes 8
```

Lưu ý về môi trường (Windows / PowerShell):
- `run.sh` là script bash; trên Windows bạn có thể dùng WSL hoặc gọi trực tiếp `python training_decoder.py ...` trong PowerShell.

Cách debug nhanh khi gặp lỗi channel/shape
------------------------------------------
1. Kiểm tra conv đầu của encoder:
   - In `in_channels`, `out_channels`, `groups` của conv đầu (nếu có) để biết encoder mong input bao nhiêu channel.
2. Kiểm tra xem có adapter 1x1 nào được tạo hay không và in shape của `adapter.weight`.
3. Chạy dummy forward (ví dụ với một tensor zeros) để in kích thước các intermediate feature maps trước khi chạy training thật.

Tài nguyên / ghi chú
---------------------
- Nếu bạn muốn mình hỗ trợ mở rộng SegNet cho một backbone mobile cụ thể (ví dụ MobileViT phiên bản X), mình có thể:
  - kiểm tra cấu trúc encoder cụ thể, viết mapping tĩnh chính xác cho phiên bản đó, hoặc
  - implement hook-based extractor (general) và test kỹ để tối ưu chọn feature maps.

Kết luận
--------
SegNet trong repo này là thiết kế cho ResNet; sử dụng DeepLabV3Plus khi bạn muốn backbone mobile sẽ an toàn và ít rủi ro hơn. Nếu bạn muốn mình triển khai hỗ trợ SegNet cho backbone mobile, mình sẽ làm theo yêu cầu và test kỹ trước khi merge.
