# CHƯƠNG 1: GIỚI THIỆU

## 1.1. Đặt vấn đề và mục tiêu nghiên cứu

### 1.1.1. Bối cảnh và nhu cầu thực tiễn

Thị trường tài chính số — bao gồm thị trường tiền mã hóa (cryptocurrency) giao dịch giao ngay (spot) và thị trường kim loại quý (điển hình là cặp XAU/USD) — trong thập kỷ vừa qua đã chuyển dịch từ một sân chơi dành riêng cho các định chế tài chính sang một môi trường đầu tư đại chúng hóa (democratized investing), nơi bất kỳ cá nhân nào cũng có thể tiếp cận thị trường 24/7 thông qua các sàn giao dịch trực tuyến. Đặc điểm nổi bật của các thị trường này — khác biệt căn bản so với thị trường chứng khoán truyền thống — là tính liên tục (không có phiên đóng/mở), độ biến động (volatility) cao, và tốc độ lan truyền thông tin cực nhanh thông qua mạng xã hội và các kênh tin tức phi chính thống. Hệ quả trực tiếp của các đặc điểm này là khối lượng dữ liệu thị trường (market data) mà một nhà đầu tư cần xử lý để ra quyết định tăng lên theo cấp số nhân theo thời gian thực, trong khi năng lực xử lý thông tin và thời gian phản ứng của con người là hữu hạn.

Song song với sự bùng nổ của thị trường, lĩnh vực trí tuệ nhân tạo — cụ thể là các mô hình học sâu (deep learning) chuyên biệt cho dữ liệu chuỗi thời gian (time-series) và các mô hình ngôn ngữ lớn (Large Language Model — LLM) — đã đạt đến một ngưỡng trưởng thành đủ để được ứng dụng vào bài toán dự báo tài chính một cách có cơ sở khoa học, thay vì chỉ dừng ở mức các chỉ báo kỹ thuật (technical indicator) cổ điển như Đường trung bình động (Moving Average), Chỉ số sức mạnh tương đối (RSI) hay MACD — vốn chỉ phản ánh trạng thái quá khứ của giá mà không có khả năng ngoại suy xác suất cho tương lai kèm theo cận tin cậy (confidence interval) định lượng.

Xuất phát từ hai xu hướng hội tụ nói trên, đề tài **AetherForecast** được đặt ra nhằm xây dựng một nền tảng dự báo tài chính (financial forecasting platform) tích hợp mô hình học máy chuỗi thời gian dạng nền tảng (Foundation Model) với một hệ thống tác nhân trí tuệ nhân tạo (Multi-Agent System) đóng vai trò cố vấn giao dịch, được trình bày thông qua một giao diện đầu cuối giao dịch (trading terminal) chuyên nghiệp theo thời gian thực.

### 1.1.2. Giới hạn của nhà đầu tư cá nhân (retail trader)

So với các định chế tài chính chuyên nghiệp (quỹ đầu tư định lượng, bàn giao dịch tự doanh của ngân hàng đầu tư), nhà đầu tư cá nhân (retail trader) phải đối mặt với ba giới hạn mang tính cấu trúc:

**Giới hạn về hạ tầng phân tích định lượng.** Các quỹ định lượng vận hành các hệ thống dự báo nội bộ dựa trên mô hình học máy được huấn luyện trên hạ tầng tính toán GPU quy mô lớn, trong khi nhà đầu tư cá nhân chỉ có thể tiếp cận các chỉ báo kỹ thuật miễn phí có sẵn trên nền tảng biểu đồ phổ thông. Khoảng cách về năng lực phân tích (analytical capability gap) này là nguyên nhân gốc rễ dẫn đến bất đối xứng thông tin trên thị trường.

**Giới hạn về khả năng xử lý đa chiều thông tin trong thời gian thực.** Một quyết định giao dịch có cơ sở đòi hỏi tổng hợp đồng thời dữ liệu giá (price action), bối cảnh vĩ mô/tin tức (sentiment), và quản trị rủi ro (position sizing, stop-loss). Con người xử lý tuần tự (sequential cognitive processing) trong khi thị trường vận động song song và liên tục — dẫn đến độ trễ quyết định (decision latency) và thiên kiến cảm xúc (emotional bias) khi giao dịch dưới áp lực biến động giá mạnh.

**Giới hạn về khả năng diễn giải rủi ro định lượng.** Ngay cả khi có được một dự báo giá, nhà đầu tư cá nhân thường thiếu công cụ để chuyển hóa dự báo đó thành một kế hoạch giao dịch có cấu trúc — bao gồm vùng vào lệnh, điểm cắt lỗ, tỷ lệ rủi ro/lợi nhuận (risk/reward ratio) và kích thước vị thế (position sizing) phù hợp với khẩu vị rủi ro cá nhân.

### 1.1.3. Mục tiêu nghiên cứu

Trên cơ sở các giới hạn đã nêu, đề tài xác lập các mục tiêu nghiên cứu và triển khai như sau:

1. Xây dựng một hệ thống thu thập và xử lý dữ liệu thị trường theo thời gian thực, có khả năng phục vụ đồng thời hiển thị biểu đồ trực quan và làm đầu vào cho mô hình dự báo.
2. Ứng dụng một mô hình học máy chuỗi thời gian dạng nền tảng (Foundation Model), có khả năng dự báo không cần huấn luyện lại theo từng tài sản (zero-shot forecasting), nhằm rút ngắn vòng đời phát triển mô hình so với cách tiếp cận huấn luyện mô hình chuyên biệt cho từng cặp giao dịch.
3. Thiết kế một hệ thống đa tác nhân trí tuệ nhân tạo (Multi-Agent System) đóng vai trò một "hội đồng cố vấn" (AI Council), có khả năng chuyển hóa kết quả dự báo định lượng thành một khuyến nghị giao dịch có cấu trúc, kèm theo lập luận minh bạch (explainable reasoning).
4. Triển khai toàn bộ hệ thống trên hạ tầng đám mây theo mô hình hạ tầng dưới dạng mã nguồn (Infrastructure as Code), đảm bảo các thuộc tính phi chức năng tối thiểu về bảo mật, khả năng vận hành liên tục, và khả năng quan sát hệ thống (observability) cần thiết cho một dịch vụ công khai trên Internet.

Cần nhấn mạnh ngay từ đầu chương này — và sẽ được làm rõ hơn ở mục 1.3 — rằng mục tiêu của đề tài là xây dựng một **hệ thống cố vấn và dự báo** (advisory & forecasting system), không phải một **hệ thống giao dịch tự động** (automated trading engine) có khả năng tự đặt lệnh và quản lý vốn không giám sát.

## 1.2. Thách thức đặt ra

Việc hiện thực hóa các mục tiêu ở mục 1.1.3 đặt ra ba nhóm thách thức kỹ thuật trọng yếu, mỗi nhóm tương ứng với một quyết định kiến trúc cốt lõi được trình bày chi tiết ở Chương 2 và Chương 3.

### 1.2.1. Thách thức xử lý dữ liệu thời gian thực

Dữ liệu nến (candlestick/OHLCV — Open, High, Low, Close, Volume) của thị trường tiền mã hóa được sinh ra liên tục với tần suất cao nhất lên đến từng phút, và một hệ thống hiển thị biểu đồ chuyên nghiệp cần đảm bảo đồng thời hai thuộc tính vốn dĩ đối lập nhau:

- **Tính tức thời (low-latency streaming):** biểu đồ phải phản ánh biến động giá gần như ngay lập tức, đòi hỏi một kênh truyền dữ liệu hai chiều, thường trực (persistent bidirectional channel) thay vì mô hình yêu cầu/phản hồi (request/response) truyền thống.
- **Tính đầy đủ của lịch sử (historical completeness):** để tính toán các chỉ báo kỹ thuật (RSI 14 chu kỳ, MACD 12-26-9) và làm đầu vào ngữ cảnh (context window) cho mô hình dự báo, hệ thống cần truy xuất một khối lượng lớn dữ liệu lịch sử được lưu trữ có cấu trúc, truy vấn hiệu quả theo tài sản và khung thời gian (timeframe).

Thách thức đặt ra là phải dung hòa hai yêu cầu này mà không đánh đổi trải nghiệm người dùng: dữ liệu tức thời phải "vá" (patch) liền mạch vào dữ liệu lịch sử đã tải, kể cả trong tình huống mất kết nối tạm thời và cần đồng bộ lại (backfill) sau khi kết nối được khôi phục.

### 1.2.2. Thách thức "ảo giác" (hallucination) của trí tuệ nhân tạo

Khi sử dụng mô hình ngôn ngữ lớn (LLM) làm thành phần suy luận cho "hội đồng cố vấn" AI, đề tài phải đối mặt trực diện với hiện tượng **ảo giác** (hallucination) — xu hướng mô hình ngôn ngữ tạo sinh ra thông tin có vẻ hợp lý về mặt ngôn ngữ nhưng không có cơ sở thực tế hoặc không nhất quán với dữ liệu đầu vào. Trong bối cảnh một hệ thống tư vấn tài chính, hallucination không đơn thuần là một lỗi hiển thị mà mang rủi ro tài chính trực tiếp cho người dùng cuối nếu một con số (vùng giá vào lệnh, điểm cắt lỗ, đòn bẩy) được mô hình "bịa" ra mà không neo (anchor) vào dữ liệu giá thực tế.

Thách thức này chi phối trực tiếp hai quyết định thiết kế được trình bày ở Chương 2 và Chương 3: (i) tách bạch rõ ràng vai trò suy luận xác suất định lượng (do mô hình Chronos-2 đảm nhiệm, cho ra cận tin cậy có thể kiểm chứng thống kê) khỏi vai trò diễn giải ngôn ngữ tự nhiên (do các tác nhân LLM đảm nhiệm), và (ii) ràng buộc đầu ra của hội đồng AI theo một lược đồ dữ liệu có cấu trúc (structured output schema) thay vì văn bản tự do, để mọi con số tài chính then chốt (entry, stop_loss, take_profit, leverage) đều được kiểm tra tính hợp lệ ở tầng ứng dụng trước khi hiển thị cho người dùng.

### 1.2.3. Thách thức bảo mật cho hệ thống công khai (public-facing)

Vì AetherForecast là một dịch vụ được công bố công khai trên Internet (public-facing service), bề mặt tấn công (attack surface) của hệ thống trải rộng trên nhiều lớp: giao diện người dùng, API xử lý nghiệp vụ, kênh dữ liệu thời gian thực, kho lưu trữ dữ liệu/mô hình trên đám mây, và hạ tầng máy chủ vận hành liên tục. Thách thức đặt ra không chỉ là ngăn chặn các hình thức tấn công phổ biến (giả mạo phiên đăng nhập, chèn mã độc, từ chối dịch vụ) mà còn phải cân bằng giữa mức độ phòng thủ và ràng buộc thực tế của một đề tài có phạm vi và nguồn lực triển khai giới hạn — tức là phải xác định rõ **những lớp phòng thủ nào đã được hiện thực hóa trong phạm vi đề tài này, và những lớp phòng thủ nào được xác định là hướng mở rộng** (được trình bày minh bạch ở mục 2.2.4 và Chương 5). Cách tiếp cận này phản ánh đúng tinh thần của mô hình bảo mật theo chiều sâu (Defense-in-Depth) — không có hệ thống nào "an toàn tuyệt đối", mà chỉ có các lớp phòng thủ được ưu tiên triển khai theo mức độ rủi ro và giá trị tài sản cần bảo vệ.

## 1.3. Nội dung và phạm vi thực hiện

### 1.3.1. Phạm vi thị trường và tài sản

Đề tài giới hạn phạm vi thị trường thực nghiệm ở hai nhóm tài sản đại diện cho hai lớp thị trường có đặc tính biến động khác nhau:

- **Thị trường giao ngay tiền mã hóa (Binance Spot):** các cặp giao dịch phổ biến (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, ...), đại diện cho lớp tài sản có độ biến động cao, giao dịch liên tục 24/7, không có giờ đóng/mở phiên.
- **Thị trường kim loại quý (XAUUSD — vàng/USD):** đại diện cho lớp tài sản truyền thống, có đặc tính biến động thấp hơn nhưng chịu ảnh hưởng mạnh bởi các yếu tố vĩ mô (lãi suất, chỉ số USD, địa chính trị).

Việc lựa chọn đồng thời hai lớp tài sản có đặc tính thống kê khác biệt (biến động cao/liên tục so với biến động thấp/chịu ảnh hưởng vĩ mô) nhằm mục đích kiểm chứng tính tổng quát hóa (generalization) của cách tiếp cận zero-shot forecasting bằng Foundation Model — một mô hình duy nhất, không huấn luyện lại riêng cho từng tài sản, được kỳ vọng vẫn tạo ra dự báo có ý nghĩa thống kê trên cả hai lớp tài sản.

### 1.3.2. Phạm vi chức năng — Ranh giới với hệ thống giao dịch tự động

Đây là ranh giới phạm vi quan trọng nhất của đề tài, cần được xác lập rõ ràng để tránh nhầm lẫn về bản chất hệ thống:

**AetherForecast KHÔNG phải là một trading engine** — tức là hệ thống **không** thực hiện các chức năng: (i) kết nối và đặt lệnh trực tiếp lên sàn giao dịch thông qua API khóa (API key) của người dùng; (ii) tự động quản lý vốn hoặc tự động thực thi chiến lược giao dịch mà không có sự xác nhận của con người; (iii) lưu trữ hoặc xử lý thông tin tài khoản giao dịch/API key của sàn giao dịch của người dùng.

**Phạm vi chức năng thực tế của hệ thống** dừng lại ở vai trò **cố vấn và dự báo** (advisory & forecasting): hiển thị dữ liệu thị trường thời gian thực, sinh dự báo giá xác suất có cận tin cậy, và tổng hợp một khuyến nghị giao dịch có cấu trúc (bao gồm vùng vào lệnh, điểm dừng lỗ, chốt lời, đòn bẩy đề xuất, kích thước vị thế) — toàn bộ khuyến nghị này mang tính chất thông tin tham khảo, quyết định thực thi cuối cùng (nếu có) hoàn toàn thuộc về người dùng, được thực hiện thủ công trên nền tảng giao dịch của họ, nằm ngoài phạm vi đề tài này. Việc giới hạn phạm vi theo hướng này không chỉ giảm thiểu rủi ro tài chính trực tiếp cho người dùng cuối, mà còn giảm đáng kể diện rủi ro bảo mật của hệ thống — vì hệ thống không bao giờ nắm giữ quyền truy cập vào tài sản thực của người dùng.

### 1.3.3. Đối tượng sử dụng

Hệ thống hướng đến hai nhóm đối tượng: (i) **người dùng cuối** — nhà đầu tư cá nhân cần một công cụ tổng hợp dữ liệu thị trường, dự báo định lượng, và khuyến nghị giao dịch có cấu trúc trên cùng một giao diện; (ii) **quản trị viên vận hành** (operator/DevOps) — chịu trách nhiệm triển khai, giám sát, và cập nhật phiên bản mô hình học máy của hệ thống trên hạ tầng đám mây.

## 1.4. Kết quả đạt được

Bảng 1.1 và Bảng 1.2 tổng hợp các tiêu chí chức năng và phi chức năng mà hệ thống đã hiện thực hóa được trong phạm vi đề tài, làm cơ sở đối chiếu chi tiết ở Chương 4 (Thử nghiệm) và tổng kết ở Chương 5 (Kết luận).

**Bảng 1.1 – Tiêu chí chức năng đã hiện thực hóa**

| STT | Tiêu chí chức năng | Mô tả hiện thực hóa |
|---|---|---|
| 1 | Dữ liệu thị trường thời gian thực | Truyền dữ liệu nến qua kênh WebSocket song song với truy vấn lịch sử theo trang (paginated historical fetch), hiển thị biểu đồ nến cùng chỉ báo RSI(14)/MACD(12,26,9) |
| 2 | Dự báo giá bằng Foundation Model | Sinh dự báo xác suất đa bước (multi-horizon) kèm dải tin cậy (confidence bands) và dải biến động (volatility bands) theo cơ chế zero-shot, không huấn luyện lại theo từng tài sản |
| 3 | Hội đồng cố vấn AI (AI Council) | Hệ thống đa tác nhân tuần tự tổng hợp khuyến nghị giao dịch có cấu trúc (entry, stop-loss, take-profit kép, đòn bẩy, kích thước vị thế, lập luận) từ kết quả dự báo, truyền trực tiếp theo thời gian thực (streaming) về giao diện |
| 4 | Đa ngôn ngữ | Giao diện hỗ trợ song ngữ Việt/Anh, tự động phát hiện và ghi nhớ lựa chọn ngôn ngữ giữa các phiên sử dụng |
| 5 | Xác thực người dùng | Đăng ký/đăng nhập qua dịch vụ quản lý danh tính tập trung, phiên làm việc được bảo vệ theo token |

**Bảng 1.2 – Tiêu chí phi chức năng đã hiện thực hóa**

| STT | Tiêu chí phi chức năng | Mô tả hiện thực hóa |
|---|---|---|
| 1 | Khả năng vận hành liên tục khi cập nhật | Quy trình triển khai tự động theo mô hình canary (triển khai phiên bản mới song song, kiểm tra sức khỏe dịch vụ, tự động khôi phục (rollback) phiên bản cũ nếu kiểm tra thất bại), giảm thiểu — không loại bỏ hoàn toàn — thời gian gián đoạn khi cập nhật |
| 2 | Kiểm soát truy cập hạ tầng | Máy chủ ứng dụng production không mở cổng quản trị từ xa truyền thống (SSH); toàn bộ thao tác quản trị được thực hiện qua kênh quản lý phiên có kiểm soát của nhà cung cấp dịch vụ đám mây |
| 3 | Bảo mật lưu trữ dữ liệu | Dữ liệu chuỗi thời gian và mô hình học máy được lưu trữ có mã hóa tại chỗ (encryption at rest) và bật tính năng quản lý phiên bản (versioning) |
| 4 | Kiểm soát truy cập API | Toàn bộ API nghiệp vụ yêu cầu token xác thực hợp lệ được cấp phát và kiểm chứng bởi dịch vụ quản lý danh tính; nguồn gọi API được giới hạn theo danh sách xuất xứ (origin) tường minh |
| 5 | Hiệu quả chi phí hạ tầng tính toán | Tài nguyên tính toán GPU cho huấn luyện được cấp phát đàn hồi theo nhu cầu (elastic on-demand/spot compute) thay vì duy trì thường trực |

---

# CHƯƠNG 2: PHƯƠNG PHÁP THỰC HIỆN

## 2.1. Khảo sát các hệ thống tương tự

Trước khi xác lập kiến trúc kỹ thuật, đề tài tiến hành khảo sát ba nhóm hệ thống tương tự đang tồn tại trên thị trường, nhằm định vị rõ khoảng trống (gap) mà AetherForecast hướng đến lấp đầy.

**Nhóm nền tảng biểu đồ và cộng đồng ý tưởng giao dịch** (ví dụ điển hình: TradingView) cung cấp công cụ vẽ biểu đồ chuyên nghiệp, thư viện chỉ báo kỹ thuật phong phú, và một mạng xã hội để người dùng chia sẻ ý tưởng giao dịch (trading ideas) dưới dạng chú thích thủ công trên biểu đồ. Hạn chế của nhóm này là **không có thành phần dự báo định lượng nội tại** — mọi phân tích dự báo đều do con người thực hiện thủ công và chia sẻ dưới dạng văn bản/hình ảnh, không có cơ sở thống kê có thể kiểm chứng hay tái lập.

**Nhóm API dữ liệu thị trường của các sàn giao dịch** (ví dụ: Binance API) cung cấp dữ liệu giá thô (raw OHLCV, sổ lệnh, giao dịch khớp) với độ trễ thấp và độ tin cậy cao, nhưng **không đi kèm bất kỳ lớp phân tích hay dự báo nào** — đây thuần túy là hạ tầng dữ liệu, đòi hỏi người dùng tự xây dựng toàn bộ tầng phân tích phía trên.

**Nhóm bot giao dịch tự động và sao chép lệnh (copy-trading)** (ví dụ điển hình: 3Commas, Pionex, Cornix) tập trung vào việc **thực thi** chiến lược giao dịch đã được định nghĩa trước (rule-based) hoặc sao chép lệnh của một trader khác, không có thành phần dự báo bằng mô hình học máy nền tảng (Foundation Model) và thường vận hành theo cơ chế "hộp đen" (black-box) — người dùng không được cung cấp lập luận giải thích (explainability) cho quyết định giao dịch được sinh ra.

Đối chiếu ba nhóm hệ thống trên, khoảng trống được xác định là: **chưa có một nền tảng nào kết hợp đồng thời (i) dự báo xác suất bằng mô hình nền tảng chuyên biệt cho chuỗi thời gian tài chính, (ii) một lớp diễn giải bằng hệ thống đa tác nhân AI tạo ra khuyến nghị có cấu trúc kèm lập luận minh bạch, và (iii) trình bày toàn bộ trên một giao diện đầu cuối giao dịch thời gian thực duy nhất** — đây chính là định vị giá trị cốt lõi của AetherForecast.

`[CHÚ THÍCH: CHÈN BẢNG SO SÁNH CÁC HỆ THỐNG TƯƠNG TỰ (TradingView / Binance API / Bot Copy-trading / AetherForecast) THEO CÁC TIÊU CHÍ: DỰ BÁO ĐỊNH LƯỢNG, LẬP LUẬN AI, GIAO DIỆN THỜI GIAN THỰC, PHẠM VI THỰC THI LỆNH TẠI ĐÂY]`

## 2.2. Cơ sở lý thuyết

### 2.2.1. Foundation Model cho chuỗi thời gian — Chronos-2

**Khái niệm Foundation Model.** Một Foundation Model là một mô hình học sâu quy mô lớn được huấn luyện trước (pretrained) trên một khối lượng dữ liệu đa dạng và tổng quát, sau đó có thể được áp dụng trực tiếp — hoặc chỉ cần tinh chỉnh (fine-tune) tối thiểu — cho nhiều bài toán hạ nguồn (downstream task) khác nhau mà không cần huấn luyện lại từ đầu. Khái niệm này vốn phổ biến trong xử lý ngôn ngữ tự nhiên (các mô hình GPT, T5) và gần đây được mở rộng sang lĩnh vực dữ liệu chuỗi thời gian, hình thành nhóm mô hình được gọi là **Time-Series Foundation Model**.

**Chronos và cơ chế token hóa chuỗi thời gian.** Đề tài sử dụng họ mô hình **Chronos** (cụ thể là phiên bản **Chronos-2**, mã định danh `amazon/chronos-2`, với phương án dự phòng — fallback — là `amazon/chronos-t5-large`) do Amazon Science phát triển. Điểm khác biệt căn bản của Chronos so với các mô hình chuỗi thời gian cổ điển (ARIMA, Prophet, hay các mạng hồi quy LSTM/GRU huấn luyện riêng cho từng chuỗi) nằm ở cách tiếp cận: Chronos **token hóa** (tokenize) giá trị chuỗi thời gian liên tục thành một chuỗi các token rời rạc, thông qua hai bước biến đổi — (i) **chuẩn hóa theo tỷ lệ** (scaling), đưa các giá trị về một miền chung bất biến với đơn vị đo và biên độ giá trị tuyệt đối của từng tài sản cụ thể, và (ii) **lượng tử hóa** (quantization), ánh xạ giá trị đã chuẩn hóa vào một tập hữu hạn các "bin" rời rạc — về bản chất là tái sử dụng kiến trúc mô hình ngôn ngữ dạng chuỗi-sang-chuỗi (sequence-to-sequence, cụ thể là kiến trúc T5/Transformer encoder-decoder) vốn được thiết kế cho token văn bản, để mô hình hóa **phân phối xác suất của bước giá tiếp theo** như một bài toán dự đoán token tiếp theo. Nhờ được huấn luyện trước trên một tập hợp cực lớn các chuỗi thời gian thuộc nhiều lĩnh vực khác nhau (không giới hạn ở dữ liệu tài chính), mô hình học được các mẫu hình thống kê tổng quát (xu hướng, tính chu kỳ, độ biến động cụm — volatility clustering) có thể khái quát hóa sang một chuỗi thời gian hoàn toàn mới **mà không cần huấn luyện lại** — đây chính là năng lực **dự báo zero-shot** (zero-shot forecasting) mà đề tài khai thác trực tiếp: cùng một mô hình được nạp một lần (xem mục 2.4.1 về quy trình MLOps) có thể tạo dự báo có ý nghĩa thống kê cho cả cặp BTCUSDT lẫn XAUUSD mà không cần một chu trình huấn luyện chuyên biệt cho từng tài sản.

**Đầu ra xác suất và cận tin cậy.** Vì bản chất đầu ra của Chronos là một **phân phối xác suất** trên không gian giá trị tương lai (chứ không phải một con số dự báo điểm — point forecast duy nhất), hệ thống có thể trích xuất trực tiếp các phân vị (quantile) khác nhau của phân phối đó để tạo thành **dải tin cậy** (confidence bands) và **dải biến động** (volatility bands) hiển thị trên biểu đồ — đây là một thuộc tính có giá trị thực tiễn rất lớn đối với bài toán tài chính, vì nó cho phép định lượng rõ ràng mức độ không chắc chắn của dự báo, thay vì trình bày một con số dự báo "chắc chắn" gây hiểu lầm cho người dùng.

### 2.2.2. Luận chứng lựa chọn: Học máy chuỗi thời gian (Chronos-2) so với Học tăng cường (Reinforcement Learning)

Đây là một trong những quyết định kiến trúc nền tảng nhất của đề tài, cần được luận giải tường minh vì Học tăng cường (Reinforcement Learning — RL) là một hướng tiếp cận phổ biến trong tài liệu học thuật về giao dịch định lượng bằng AI, và một cách trực giác, RL có vẻ "phù hợp" hơn cho bài toán giao dịch vì bản chất tuần tự ra quyết định của nó. Tuy nhiên, đề tài đã **cân nhắc và chủ động không lựa chọn** hướng tiếp cận RL, dựa trên các luận điểm sau:

**Thứ nhất, RL đòi hỏi một môi trường tương tác trực tiếp (interactive environment) để sinh tín hiệu phản hồi (reward signal).** Bản chất của một tác tử RL (RL agent) là học chính sách (policy) tối ưu thông qua thử-sai (trial-and-error) trong một vòng lặp hành động — quan sát — phần thưởng (action–observation–reward loop). Áp dụng vào bài toán giao dịch, điều này đồng nghĩa với việc tác tử phải **thực sự đặt lệnh** (hoặc mô phỏng đặt lệnh với đầy đủ chi phí giao dịch, trượt giá — slippage, và độ trễ khớp lệnh) để tính toán được phần thưởng (lợi nhuận/thua lỗ thực tế) sau mỗi hành động, rồi lặp lại hàng triệu lượt để chính sách hội tụ. Yêu cầu này đặt ra hai vấn đề nghiêm trọng: (i) nó vượt hoàn toàn ra ngoài phạm vi một hệ thống **cố vấn** như đã xác lập ở mục 1.3.2 — biến hệ thống từ một công cụ tư vấn thành một **cỗ máy tự thực thi giao dịch** với vốn thật, và (ii) ngay cả khi mô phỏng trên môi trường backtest lịch sử thay vì vốn thật, một tác tử RL huấn luyện trên dữ liệu quá khứ vẫn tiềm ẩn rủi ro **quá khớp** (overfitting) nghiêm trọng với các mẫu hình đặc thù của giai đoạn lịch sử đó, và **không đảm bảo tổng quát hóa** sang chế độ thị trường (market regime) chưa từng xuất hiện trong dữ liệu huấn luyện.

**Thứ hai, rủi ro vốn (capital risk) trong giai đoạn huấn luyện và suy luận trực tuyến (online learning) là không thể chấp nhận được đối với phạm vi đề tài.** Nhiều kiến trúc RL cho giao dịch (ví dụ các thuật toán họ Policy Gradient như PPO, hay Actor-Critic như DDPG) đạt hiệu năng tốt nhất khi được phép tiếp tục học trực tuyến (online fine-tuning) trên dữ liệu thị trường mới nhất — nhưng điều này có nghĩa là chính sách giao dịch của hệ thống **có thể thay đổi hành vi mà không có sự kiểm soát và xác nhận của con người** ngay trong quá trình vận hành thực tế, một đặc tính hoàn toàn đối lập với triết lý "con người luôn là người ra quyết định cuối cùng" đã xác lập ở mục 1.3.2.

**Thứ ba, tính phi dừng (non-stationarity) và nhiễu (noise) cao của dữ liệu tài chính khiến không gian trạng thái–hành động (state-action space) của bài toán RL trở nên cực kỳ khó tối ưu ổn định.** Không giống các bài toán RL kinh điển có môi trường xác định hoặc bán xác định (trò chơi, robot điều khiển), thị trường tài chính là một hệ thống **phi dừng** (thống kê phân phối giá thay đổi theo thời gian, chế độ thị trường luân chuyển giữa xu hướng/đi ngang/biến động mạnh) và có tỷ lệ tín hiệu/nhiễu (signal-to-noise ratio) rất thấp. Nhiều nghiên cứu thực nghiệm trong lĩnh vực đã ghi nhận rằng các tác tử RL huấn luyện trên dữ liệu tài chính lịch sử có xu hướng học được các chiến lược khai thác quá mức (exploit) những đặc thù nhiễu ngẫu nhiên của giai đoạn huấn luyện — biểu hiện bên ngoài là hiệu năng backtest rất ấn tượng nhưng suy giảm mạnh (hoặc thất bại hoàn toàn) khi triển khai thực tế (live trading), một hiện tượng được gọi là "backtest overfitting".

**Ngược lại, hướng tiếp cận Học máy dự báo chuỗi thời gian bằng Foundation Model (Chronos-2) giải quyết trực tiếp cả ba vấn đề trên:** (i) bài toán được đặt lại đúng bản chất là **dự báo có giám sát** (supervised forecasting) trên dữ liệu lịch sử đã biết nhãn thực tế (giá trị tương lai đã xảy ra), không đòi hỏi một môi trường tương tác trực tiếp hay tính toán phần thưởng từ việc thực thi giao dịch thật; (ii) mô hình được nạp và cố định (frozen) tại thời điểm suy luận — không có cơ chế tự thay đổi hành vi trong quá trình vận hành mà không qua kiểm soát của con người (xem thêm mục 2.4.1); và (iii) vì đây là mô hình nền tảng đã huấn luyện trước trên tập dữ liệu khổng lồ và đa dạng lĩnh vực, khả năng khái quát hóa của nó vượt trội so với một tác tử RL chỉ được huấn luyện (hoặc tự học) trên riêng dữ liệu lịch sử của thị trường tài chính mục tiêu — đồng thời đầu ra của mô hình luôn đi kèm **cận tin cậy định lượng có thể kiểm chứng thống kê** (như trình bày ở mục 2.2.1), thay vì một hành động rời rạc "mua/bán/giữ" không có cơ sở xác suất tường minh như đầu ra điển hình của một chính sách RL.

Tóm lại, việc lựa chọn Chronos-2 thay vì RL là một quyết định có chủ đích nhằm **tối đa hóa độ an toàn và khả năng diễn giải, chấp nhận đánh đổi khả năng tối ưu hóa trực tiếp hàm lợi nhuận** mà một tác tử RL lý thuyết có thể hứa hẹn — một đánh đổi hoàn toàn phù hợp với bản chất **cố vấn, không thực thi** của hệ thống.

### 2.2.3. Kiến trúc hệ thống đa tác nhân (Multi-Agent System) cho tư vấn giao dịch

**Cơ sở lý thuyết: hai mô hình điều phối tác nhân.** Trong lĩnh vực hệ thống đa tác nhân dựa trên mô hình ngôn ngữ lớn (LLM Multi-Agent System), tồn tại hai lớp kiến trúc điều phối chính mà đề tài đã khảo sát và cân nhắc:

- **Kiến trúc tuần tự (Sequential Pipeline):** mỗi tác nhân đảm nhiệm một vai trò chuyên biệt và chuyển giao kết quả cho tác nhân kế tiếp theo một luồng xử lý một chiều, tuyến tính, không có cơ chế phản hồi ngược (feedback loop) hay xét lại quyết định. Framework tiêu biểu cho mô hình này là **CrewAI**, tổ chức các tác nhân thành một `Crew` thực thi theo `Process` (tuần tự — sequential, hoặc phân cấp — hierarchical).
- **Kiến trúc đồ thị lặp (Graph-based Iterative Architecture):** các tác nhân được tổ chức thành các đỉnh (node) của một đồ thị trạng thái (state graph), cho phép luồng xử lý rẽ nhánh có điều kiện và **quay lại** (loop back) một tác nhân trước đó để xét lại/tranh biện (debate) kết quả trước khi đi đến kết luận cuối cùng. Framework tiêu biểu cho mô hình này là **LangGraph**, trong đó một mẫu hình phổ biến là bổ sung một tác nhân đóng vai trò **"người phản biện"** (Devil's Advocate) — có nhiệm vụ chủ động tìm kiếm điểm yếu, giả định chưa được kiểm chứng, hoặc rủi ro bị bỏ sót trong kết luận của các tác nhân khác, buộc hệ thống phải xét lại (revise) trước khi chốt khuyến nghị cuối cùng.

**Đánh đổi giữa hai kiến trúc.** Mô hình đồ thị lặp có ưu điểm lý thuyết rõ rệt về khả năng **tự sửa lỗi** (self-correction) và giảm thiểu thiên kiến của một tác nhân đơn lẻ (single-agent bias), nhờ cơ chế tranh biện nội bộ buộc các giả định phải được kiểm chứng chéo trước khi đi đến kết luận cuối. Tuy nhiên, đánh đổi của mô hình này là: (i) **độ trễ tổng thể** (latency) tăng lên đáng kể và khó dự đoán chính xác (do số vòng lặp tranh biện không cố định, phụ thuộc điều kiện hội tụ), (ii) **chi phí token** (do gọi LLM nhiều lượt hơn) khó giới hạn chặn trên (bounded) một cách tường minh, và (iii) độ phức tạp quản lý trạng thái (state management) giữa các vòng lặp tăng lên đáng kể, kéo theo rủi ro lỗi vận hành cao hơn ở giai đoạn triển khai ban đầu của một hệ thống.

**Kiến trúc đã hiện thực hóa.** Trong phạm vi đề tài này, "Hội đồng cố vấn AI" (AI Council) được hiện thực hóa theo **kiến trúc tuần tự bằng CrewAI**, với ba tác nhân chuyên biệt được thực thi theo đúng một lượt duy nhất (`Process.sequential`), không có vòng lặp xét lại:

1. **Nhà phân tích định lượng (Quant Analyst)** — tiếp nhận đầu ra xác suất từ mô hình Chronos-2 (dự báo giá, dải tin cậy, dải biến động) cùng dữ liệu ngữ cảnh thị trường (sentiment) và diễn giải thành một nhận định xu hướng.
2. **Nhà quản trị rủi ro (Risk Manager)** — tiếp nhận nhận định của Nhà phân tích định lượng, tính toán và đề xuất các thông số quản trị rủi ro cụ thể: vùng vào lệnh, điểm dừng lỗ, tỷ lệ rủi ro/lợi nhuận, kích thước vị thế, đòn bẩy đề xuất.
3. **Thẩm phán thực thi (Execution Judge)** — tổng hợp nhận định của hai tác nhân trước, đưa ra quyết định cuối cùng (LONG / SHORT / HOLD) cùng lập luận tổng hợp, đảm bảo tính nhất quán logic giữa nhận định xu hướng và thông số rủi ro trước khi trả về kết quả có cấu trúc cho tầng ứng dụng.

Việc lựa chọn kiến trúc tuần tự cho giai đoạn hiện tại của đề tài dựa trên các lý do thực tiễn: (i) độ trễ có thể dự đoán và giới hạn trên tường minh (số lượt gọi LLM cố định = số tác nhân), phù hợp với yêu cầu truyền kết quả theo thời gian thực (streaming) qua kênh Server-Sent Events về giao diện người dùng như trình bày ở mục 2.3.1; (ii) độ phức tạp triển khai và khả năng gỡ lỗi (debuggability) thấp hơn đáng kể, phù hợp với giai đoạn xây dựng nền tảng ban đầu của hệ thống; và (iii) luồng xử lý tuyến tính giúp việc ràng buộc đầu ra theo lược đồ có cấu trúc (structured schema, xem mục 1.2.2) được thực hiện đơn giản và tin cậy hơn.

**Định hướng kiến trúc mở rộng.** Kiến trúc đồ thị lặp kèm tác nhân phản biện (Devil's Advocate) theo mô hình LangGraph — bao gồm một vòng lặp tranh biện chủ động giữa các tác nhân trước khi chốt khuyến nghị cuối cùng — đã được khảo sát và đánh giá là một hướng nâng cấp có giá trị học thuật và thực tiễn rõ ràng, nhưng được xác định là **nằm ngoài phạm vi hiện thực hóa của đề tài này** và được đưa vào định hướng phát triển (xem Chương 5, mục 5.3), vì lý do cân bằng giữa lợi ích lý thuyết và chi phí độ trễ/độ phức tạp vận hành như đã phân tích ở trên.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ SO SÁNH KIẾN TRÚC TUẦN TỰ (SEQUENTIAL — ĐÃ TRIỂN KHAI) VÀ KIẾN TRÚC ĐỒ THỊ LẶP (GRAPH-BASED DEBATE — ĐỊNH HƯỚNG MỞ RỘNG) TẠI ĐÂY]`

### 2.2.4. DevSecOps và mô hình bảo mật theo chiều sâu (Defense-in-Depth)

**Cơ sở lý thuyết.** DevSecOps là triết lý tích hợp thực hành bảo mật vào mọi giai đoạn của vòng đời phát triển và vận hành phần mềm (thay vì xử lý bảo mật như một bước kiểm tra bổ sung sau cùng — "shift-left security"), kết hợp với nguyên lý **Bảo mật theo chiều sâu** (Defense-in-Depth): thay vì dựa vào một lớp phòng thủ duy nhất, hệ thống được bảo vệ bởi nhiều lớp độc lập, sao cho một lớp bị xuyên thủng không dẫn đến toàn bộ hệ thống bị xâm phạm. Các lớp phòng thủ điển hình của một hệ thống công khai trên nền tảng đám mây bao gồm (từ ngoài vào trong): (i) lớp biên mạng/CDN với tường lửa ứng dụng web (Web Application Firewall — WAF) lọc lưu lượng độc hại trước khi đến máy chủ gốc; (ii) lớp mạng nội bộ với nhóm bảo mật (security group) giới hạn cổng và nguồn kết nối; (iii) lớp quản lý truy cập quản trị, ưu tiên các kênh quản lý phiên có kiểm soát (session manager) thay vì giao thức truy cập từ xa truyền thống (SSH) vốn đòi hỏi quản lý khóa và mở cổng công khai; (iv) lớp xác thực và ủy quyền ở tầng ứng dụng (danh tính người dùng, token, kiểm tra CORS); (v) lớp lưu trữ dữ liệu với mã hóa tại chỗ và — ở mức bảo vệ cao hơn — cơ chế bất biến hóa đối tượng lưu trữ (Object Lock/WORM — Write-Once-Read-Many) nhằm chống lại việc xóa/ghi đè dữ liệu ngay cả khi thông tin đăng nhập quản trị bị xâm phạm; và (vi) lớp toàn vẹn chuỗi cung ứng phần mềm (supply chain integrity) — đặc biệt quan trọng khi hệ thống có tải và thực thi mã nguồn từ một mô hình học máy bên thứ ba (như tùy chọn `trust_remote_code=True` được phân tích ở mục 2.2.1), đòi hỏi một cơ chế xác minh (ví dụ đối chiếu chữ ký số hoặc mã băm với một danh sách cho phép — allow-list — đã được xác thực trước) trước khi cho phép mã nguồn đó được thực thi.

**Đối chiếu với hiện trạng đã triển khai.** Trong phạm vi đề tài này, các lớp phòng thủ sau đã được hiện thực hóa và sẽ được kiểm chứng chi tiết ở Chương 4:

- **Kiểm soát truy cập quản trị hạ tầng:** máy chủ ứng dụng chính không mở cổng quản trị từ xa truyền thống (port 22/SSH); mọi thao tác triển khai và quản trị được thực hiện thông qua kênh quản lý phiên có kiểm soát của nhà cung cấp dịch vụ đám mây, loại bỏ hoàn toàn nhu cầu quản lý và phân phối khóa truy cập từ xa cho máy chủ đó.
- **Kiểm soát nguồn gọi API (CORS)** và **xác thực token (JWT)** thông qua dịch vụ quản lý danh tính tập trung, với việc kiểm chứng bên phát hành (issuer) và bên nhận (audience/client) của token được thực hiện tường minh ở tầng phụ thuộc (dependency) của khung ứng dụng API.
- **Bảo vệ dữ liệu lưu trữ** thông qua mã hóa tại chỗ và quản lý phiên bản (versioning) đối tượng lưu trữ, cho phép khôi phục về phiên bản trước trong trường hợp dữ liệu bị ghi đè ngoài ý muốn.
- **Kiểm tra tính hợp lệ đầu vào** (input validation) ở tầng lược đồ dữ liệu (schema) của khung ứng dụng API cho các tham số nghiệp vụ trọng yếu.
- **Quy trình triển khai có kiểm tra sức khỏe và tự động khôi phục** (canary deployment với health-check và automatic rollback), giảm thiểu rủi ro một phiên bản triển khai lỗi ảnh hưởng đến dịch vụ đang chạy.

**Định hướng mở rộng (được xác định minh bạch là chưa hiện thực hóa trong phạm vi đề tài).** Các lớp phòng thủ sau đã được khảo sát về mặt lý thuyết và được xác định là hướng phát triển tiếp theo, sẽ được trình bày lại ở Chương 5, mục 5.3: (i) triển khai tường lửa ứng dụng web (WAF) ở lớp biên mạng/CDN để lọc các mẫu tấn công phổ biến (SQL injection, XSS, rate-based bot traffic) trước khi lưu lượng chạm đến máy chủ gốc; (ii) bật cơ chế bất biến hóa đối tượng lưu trữ (S3 Object Lock) cho các đối tượng mô hình học máy và dữ liệu có yêu cầu lưu vết kiểm toán (audit trail) ở mức không thể chỉnh sửa; (iii) xây dựng cơ chế xác minh toàn vẹn mô hình học máy dựa trên đối chiếu mã băm với một danh sách cho phép đã ký số (signed-manifest allow-list), thay vì cơ chế mã băm hiện tại vốn chỉ phục vụ mục đích xác định dữ liệu cache đã lỗi thời hay chưa (cache-freshness), chưa đóng vai trò một cổng kiểm soát bảo mật trước khi cho phép thực thi mã nguồn từ mô hình bên thứ ba; và (iv) mở rộng cơ chế cập nhật mô hình học máy theo hướng tải lại nóng (hot-reload) thực sự không gián đoạn, thay thế cho việc mô hình hiện tại chỉ được nạp một lần duy nhất khi tiến trình dịch vụ khởi động.

Cách trình bày minh bạch này — phân định rõ ràng giữa các lớp phòng thủ đã kiểm chứng và các lớp còn là định hướng — không làm giảm giá trị học thuật của đề tài mà ngược lại, phản ánh đúng bản chất của thực hành DevSecOps trong thực tế công nghiệp: bảo mật là một quá trình cải tiến liên tục theo mức độ ưu tiên rủi ro, không phải một trạng thái "hoàn thành" tuyệt đối tại một thời điểm.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ MÔ HÌNH BẢO MẬT THEO CHIỀU SÂU (DEFENSE-IN-DEPTH), PHÂN LỚP THEO TRẠNG THÁI "ĐÃ TRIỂN KHAI" VÀ "ĐỊNH HƯỚNG MỞ RỘNG" TẠI ĐÂY]`

## 2.3. Công nghệ sử dụng

### 2.3.1. Backend: FastAPI

Tầng xử lý nghiệp vụ phía máy chủ được xây dựng trên **FastAPI** — một framework Python hiện đại cho xây dựng API, được lựa chọn dựa trên ba luận điểm kỹ thuật: (i) mô hình lập trình bất đồng bộ (asynchronous, dựa trên `async`/`await` của Python) phù hợp tự nhiên với các tác vụ I/O-bound chiếm ưu thế trong hệ thống — gọi mô hình LLM qua API bên ngoài, truy vấn kho lưu trữ đối tượng đám mây, duy trì kết nối WebSocket thường trực — cho phép một tiến trình duy nhất phục vụ đồng thời nhiều kết nối mà không cần mô hình đa luồng phức tạp; (ii) tích hợp sẵn xác thực và kiểm tra kiểu dữ liệu đầu vào/đầu ra thông qua thư viện Pydantic, cho phép định nghĩa lược đồ dữ liệu có cấu trúc (structured schema) một cách tường minh — trực tiếp phục vụ yêu cầu ràng buộc đầu ra của Hội đồng AI đã nêu ở mục 1.2.2; và (iii) hỗ trợ nguyên sinh (native) cho cả hai mô hình truyền dữ liệu thời gian thực mà hệ thống cần: **WebSocket** (dùng cho kênh truyền dữ liệu nến thời gian thực, hai chiều, thường trực) và **Server-Sent Events — SSE** (dùng cho kênh truyền một chiều, phù hợp để truyền tiến trình suy luận của Hội đồng AI về giao diện theo thời gian thực khi các tác nhân lần lượt xử lý, không cần kênh hai chiều).

### 2.3.2. Frontend: React

Giao diện người dùng được xây dựng dưới dạng ứng dụng một trang (Single-Page Application) bằng **React** kết hợp TypeScript, được lựa chọn vì mô hình lập trình theo thành phần (component-based) phù hợp tự nhiên với đặc thù một giao diện đầu cuối giao dịch — vốn được cấu thành từ nhiều khối trạng thái độc lập nhưng liên tục cập nhật đồng thời (biểu đồ giá, bảng điều khiển dự báo, bảng điều khiển Hội đồng AI, danh sách tài sản) — cùng hệ sinh thái thư viện trực quan hóa dữ liệu tài chính trưởng thành (thư viện biểu đồ nến chuyên dụng) và quản lý trạng thái phía client nhẹ (Zustand) phù hợp cho một ứng dụng có nhiều luồng cập nhật thời gian thực đồng thời mà không cần đến độ phức tạp của các giải pháp quản lý trạng thái toàn cục nặng nề hơn.

### 2.3.3. Hạ tầng đám mây AWS

Toàn bộ hạ tầng được định nghĩa và quản lý dưới dạng mã nguồn (Infrastructure as Code) trên nền tảng Amazon Web Services (AWS), theo mô hình triển khai gồm các thành phần sau:

**Kho dữ liệu Parquet phân vùng trên S3 (Partitioned Parquet Data Lake) — luận chứng lựa chọn so với cơ sở dữ liệu quan hệ (RDBMS).** Dữ liệu chuỗi thời gian (nến lịch sử) được lưu trữ dưới định dạng cột (columnar format) Apache Parquet trên Amazon S3, được phân vùng vật lý theo cấu trúc thư mục `symbol=<mã tài sản>/timeframe=<khung thời gian>/`, thay vì lưu trữ trong một cơ sở dữ liệu quan hệ truyền thống (RDBMS) như PostgreSQL/MySQL. Luận chứng cho quyết định này dựa trên đặc thù truy vấn của bài toán: (i) khối lượng truy vấn chi phối của hệ thống là **truy vấn phân tích quét theo dải thời gian** (range scan trên một tài sản/khung thời gian cụ thể để tính chỉ báo kỹ thuật hoặc làm ngữ cảnh đầu vào cho mô hình dự báo), không phải các giao dịch cập nhật đơn lẻ theo hàng (row-level transactional update) — đây chính là mẫu hình tải (workload pattern) mà định dạng cột được tối ưu hóa vượt trội so với định dạng hàng (row-oriented) của RDBMS truyền thống, vì chỉ cần đọc các cột giá trị cần thiết (open/high/low/close/volume) thay vì toàn bộ hàng; (ii) cấu trúc phân vùng vật lý theo `symbol=`/`timeframe=` cho phép **cắt tỉa phân vùng** (partition pruning) ngay ở tầng hệ thống tệp — truy vấn cho một tài sản cụ thể chỉ cần quét đúng thư mục con tương ứng, không cần quét toàn bộ tập dữ liệu; (iii) mô hình lưu trữ đối tượng (object storage) tách biệt hoàn toàn giữa tầng lưu trữ và tầng tính toán (decoupled storage/compute), cho phép mở rộng dung lượng lưu trữ độc lập với tài nguyên tính toán và tránh chi phí vận hành thường trực của một máy chủ cơ sở dữ liệu quan hệ (vốn cần duy trì instance chạy liên tục bất kể tải truy vấn thực tế); và (iv) chi phí lưu trữ trên mỗi đơn vị dữ liệu của lưu trữ đối tượng thấp hơn đáng kể so với lưu trữ khối (block storage) gắn với một instance cơ sở dữ liệu quan hệ — một yếu tố quan trọng khi khối lượng dữ liệu lịch sử nến tích lũy theo thời gian là rất lớn nhưng phần lớn dữ liệu cũ có tần suất truy vấn thấp. Đối tượng lưu trữ được cấu hình mã hóa phía máy chủ (SSE-S3) và bật quản lý phiên bản (versioning) kèm quy tắc vòng đời (lifecycle rule) tự động.

**AWS Batch cho huấn luyện mô hình.** Tác vụ huấn luyện/tinh chỉnh mô hình học máy — vốn chỉ cần chạy theo chu kỳ (không thường trực) và đòi hỏi tài nguyên GPU chuyên dụng (đề tài sử dụng dòng instance GPU `g4dn.2xlarge`) — được cấp phát thông qua AWS Batch với môi trường tính toán kết hợp instance dự phòng theo giá thị trường (spot) và theo yêu cầu (on-demand). Lựa chọn này dựa trên đặc thù tải: khối lượng công việc huấn luyện có tính **bùng nổ** (bursty — chỉ cần tài nguyên GPU trong cửa sổ thời gian chạy huấn luyện, không cần bất kỳ lúc nào khác), nên mô hình cấp phát đàn hồi theo yêu cầu công việc (job-based elastic provisioning) của AWS Batch tiết kiệm chi phí đáng kể so với phương án duy trì một instance GPU chạy thường trực.

**EC2 cho dịch vụ suy luận và huấn luyện.** Hai vai trò máy chủ tách biệt được triển khai trên EC2: một máy chủ phục vụ API ứng dụng (backend suy luận, đóng gói container, không thực hiện huấn luyện) và một máy chủ riêng biệt cho các tác vụ huấn luyện tương tác trực tiếp khi cần. Việc tách biệt hai vai trò này nhằm cô lập rủi ro và tải giữa tầng phục vụ người dùng cuối (đòi hỏi tính sẵn sàng cao, ổn định) và tầng thử nghiệm/huấn luyện mô hình (có thể chấp nhận gián đoạn ngắn hạn khi cần thử nghiệm cấu hình mới).

**CloudFront làm lớp phân phối biên (CDN).** Giao diện người dùng (ứng dụng React sau khi build) được phân phối qua CloudFront với nguồn gốc (origin) là một bucket S3 được giới hạn truy cập chỉ qua CloudFront (Origin Access Identity), tận dụng mạng lưới điểm hiện diện (Points of Presence) toàn cầu của CloudFront để giảm độ trễ tải trang cho người dùng cuối ở nhiều khu vực địa lý, đồng thời giảm tải trực tiếp lên bucket lưu trữ gốc.

**Cognito cho quản lý danh tính.** Việc đăng ký, đăng nhập, và phát hành token xác thực được ủy quyền hoàn toàn cho Amazon Cognito (User Pool), thay vì tự xây dựng hệ thống quản lý mật khẩu/phiên đăng nhập nội bộ — một quyết định giảm đáng kể diện rủi ro bảo mật liên quan đến việc lưu trữ và xử lý thông tin xác thực nhạy cảm của người dùng, đồng thời kế thừa sẵn các chính sách mật khẩu mạnh và tùy chọn xác thực đa yếu tố (MFA) do dịch vụ quản lý cung cấp.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ KIẾN TRÚC HẠ TẦNG TỔNG THỂ TRÊN AWS (CLOUDFRONT — EC2 — S3 PARQUET — AWS BATCH — COGNITO) TẠI ĐÂY]`

## 2.4. Phân tích yêu cầu

### 2.4.1. Quy trình MLOps

Vòng đời của mô hình học máy trong hệ thống được tổ chức theo một quy trình MLOps gồm các giai đoạn tách biệt rõ ràng:

1. **Huấn luyện/tinh chỉnh** trên hạ tầng AWS Batch hoặc EC2 huấn luyện chuyên dụng (mục 2.3.3), sinh ra các tệp trọng số (artifact) của mô hình.
2. **Đóng gói và công bố (publish)** artifact mô hình lên bucket S3 dành riêng cho mô hình học máy, kèm theo một tệp kê khai (manifest) mô tả phiên bản hiện hành.
3. **Nạp mô hình phía suy luận (inference loading):** dịch vụ suy luận đọc tệp kê khai, xác định vị trí artifact tương ứng, và tải mô hình về bộ nhớ cục bộ. Hệ thống sử dụng một cơ chế đối chiếu mã băm (dựa trên siêu dữ liệu đối tượng lưu trữ — khóa, ETag, kích thước, thời điểm chỉnh sửa gần nhất) để xác định liệu bản sao cục bộ đã có sẵn có còn khớp với artifact mới nhất trên S3 hay không, tránh tải lại không cần thiết nếu không có thay đổi — cần nhấn mạnh rằng cơ chế mã băm này đóng vai trò **tối ưu hóa hiệu năng nạp mô hình** (cache-freshness check), chưa phải một cơ chế **kiểm soát bảo mật/toàn vẹn** như đã phân tích ở mục 2.2.4.
4. **Suy luận:** mô hình sau khi nạp được giữ cố định (cached) trong bộ nhớ của tiến trình dịch vụ trong suốt vòng đời của tiến trình đó, phục vụ toàn bộ các yêu cầu dự báo cho đến khi tiến trình được khởi động lại (ví dụ trong một chu kỳ triển khai phiên bản mới, xem mục 2.2.4).
5. **Fallback:** nếu mô hình chính (`amazon/chronos-2`) không thể nạp thành công (do sự cố mạng, artifact lỗi, hoặc không tương thích), hệ thống chuyển sang nạp mô hình dự phòng (`amazon/chronos-t5-large`) để đảm bảo dịch vụ dự báo vẫn khả dụng ở mức tối thiểu thay vì gián đoạn hoàn toàn.

### 2.4.2. Phân tích Use Case

Hệ thống được phân tích thành hai nhóm tác nhân (actor) chính với các use case tương ứng:

**Người dùng cuối (End User):** Đăng ký/Đăng nhập; Tìm kiếm và chọn tài sản giao dịch; Xem biểu đồ nến và chỉ báo kỹ thuật theo thời gian thực; Yêu cầu sinh dự báo giá (kèm lựa chọn khung dự báo — horizon); Yêu cầu phân tích từ Hội đồng cố vấn AI; Chuyển đổi ngôn ngữ giao diện; Đăng xuất.

**Quản trị viên vận hành (Operator/DevOps):** Huấn luyện và công bố phiên bản mô hình mới; Triển khai phiên bản ứng dụng mới (kích hoạt quy trình canary deployment); Giám sát tình trạng hệ thống; Thực hiện khôi phục (rollback) khi phát hiện sự cố sau triển khai.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ USE CASE (UML) THỂ HIỆN HAI NHÓM TÁC NHÂN "NGƯỜI DÙNG CUỐI" VÀ "QUẢN TRỊ VIÊN VẬN HÀNH" CÙNG CÁC USE CASE TƯƠNG ỨNG TẠI ĐÂY]`

---

# CHƯƠNG 3: THIẾT KẾ HỆ THỐNG

## 3.1. Mô hình dữ liệu (Data Model)

Vì hệ thống sử dụng kho dữ liệu đối tượng (S3) thay cho cơ sở dữ liệu quan hệ truyền thống, mục này trình bày mô hình dữ liệu theo hai mức: mức ý niệm (phân chia các miền dữ liệu — data domain) và mức vật lý (cấu trúc phân vùng, định dạng tệp, và lược đồ tệp kê khai).

### 3.1.1. Mức ý niệm (Conceptual Level)

Toàn bộ dữ liệu của hệ thống được tổ chức thành ba miền lưu trữ tách biệt về mặt vật lý (ba bucket S3 độc lập), mỗi miền phục vụ một mục đích và một nhóm tiêu thụ (consumer) khác nhau:

- **Kho dữ liệu thị trường (Market Data Lake):** lưu trữ dữ liệu chuỗi thời gian (nến lịch sử) dưới hai dạng — dữ liệu thô phục vụ hiển thị/suy luận trực tiếp, và dữ liệu đã làm giàu đặc trưng (enriched feature data) phục vụ huấn luyện mô hình. Đây là miền dữ liệu có tần suất ghi cao nhất (được cập nhật định kỳ, xem mục 3.2.1) và tần suất đọc cao nhất (mỗi yêu cầu xem biểu đồ/dự báo của người dùng).
- **Kho lưu trữ mô hình học máy (Model Registry):** lưu trữ artifact trọng số của mô hình Chronos-2 (bản gốc và/hoặc bản đã tinh chỉnh) cùng một tệp kê khai (manifest) xác định phiên bản mô hình đang được kích hoạt cho môi trường suy luận. Miền dữ liệu này có tần suất ghi thấp (chỉ khi công bố phiên bản mô hình mới) nhưng có yêu cầu về tính nhất quán đọc rất cao (dịch vụ suy luận phải luôn đọc được đúng phiên bản đã được xác nhận).
- **Kho lưu trữ giao diện tĩnh (Frontend Hosting):** lưu trữ các tệp tĩnh đã được biên dịch (build) của ứng dụng React, được phân phối ra người dùng cuối thông qua CloudFront (mục 2.3.3), không được truy cập trực tiếp mà chỉ thông qua CDN.

Việc tách biệt vật lý ba miền dữ liệu này — thay vì gộp chung vào một bucket duy nhất — nhằm hai mục đích thiết kế: (i) **cô lập bán kính ảnh hưởng** (blast radius isolation) — một chính sách truy cập hoặc vòng đời (lifecycle) cấu hình sai trên một miền dữ liệu không ảnh hưởng đến hai miền còn lại; và (ii) **áp dụng chính sách vòng đời và quyền truy cập khác nhau phù hợp với đặc tính riêng của từng miền** — ví dụ, dữ liệu thị trường có thể áp dụng chính sách lưu trữ lạnh (cold storage tiering) cho dữ liệu cũ ít truy vấn, trong khi kho mô hình học máy cần giữ nguyên toàn bộ phiên bản trong suốt vòng đời để có thể khôi phục (rollback) về một phiên bản mô hình cũ hơn khi cần.

### 3.1.2. Mức vật lý và cấu trúc phân vùng (Physical Level & Partitioning)

**Bộ dữ liệu phục vụ (Serving Dataset).** Dữ liệu nến thô (OHLCV) dùng làm ngữ cảnh đầu vào cho mô hình dự báo được lưu trữ dưới định dạng Apache Parquet, phân vùng vật lý hai cấp theo cấu trúc:

```
s3://<market-data-bucket>/market/klines/symbol=<SYMBOL>/timeframe=<TIMEFRAME>/*.parquet
```

Ví dụ: `s3://.../market/klines/symbol=BTCUSDT/timeframe=1h/`. Cần nói rõ phạm vi sử dụng của bộ dữ liệu này: nó được đọc bởi API `/predict` (thông qua tầng truy xuất `S3ParquetClient.fetch_chart_points`) để dựng cửa sổ ngữ cảnh cho Chronos-2, theo chiến lược **ưu tiên đọc từ Parquet, kèm cơ chế bổ sung nến mới nhất trực tiếp từ Binance** nếu dữ liệu trên S3 chưa đủ mới hoặc chưa cấu hình bucket — chứ **không phải** là nguồn dữ liệu cho API hiển thị biểu đồ. API `/chart/{symbol}` phục vụ hiển thị biểu đồ trên giao diện hoạt động như một proxy đọc trực tiếp, theo thời gian thực, từ chính API công khai của Binance, hoàn toàn không đi qua kho dữ liệu S3 — một tách bạch có chủ đích giữa đường dữ liệu hiển thị (ưu tiên độ mới tuyệt đối) và đường dữ liệu suy luận (ưu tiên tính nhất quán của cửa sổ ngữ cảnh, chấp nhận đánh đổi độ trễ nhỏ để đọc từ kho đã được chuẩn hóa). Tầng truy xuất dựng đường dẫn truy vấn Parquet trực tiếp từ hai tham số `symbol` và `timeframe` của yêu cầu, cho phép **cắt tỉa phân vùng** (partition pruning) tuyệt đối — mỗi truy vấn chỉ quét đúng một thư mục con tương ứng với đúng một tài sản và một khung thời gian, không bao giờ quét chéo sang dữ liệu của tài sản khác.

**Bộ dữ liệu làm giàu đặc trưng (Enriched Feature Dataset).** Song song với bộ dữ liệu phục vụ nói trên, hệ thống duy trì một bộ dữ liệu thứ hai — được sinh ra bởi tiến trình cập nhật định kỳ trình bày ở mục 3.2.1 — có cấu trúc phân vùng bốn cấp:

```
s3://<bucket>/<prefix>/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/day=<DD>/*.parquet
```

Bộ dữ liệu này không chỉ chứa OHLCV mà còn được làm giàu thêm các trường: dữ liệu phái sinh từ thị trường hợp đồng tương lai Binance (tỷ lệ tài trợ — funding rate, khối lượng mở — open interest, tỷ lệ vị thế mua/bán — long/short ratio), dữ liệu vĩ mô (chỉ số USD — DXY, lợi suất trái phiếu kho bạc Mỹ kỳ hạn 10 năm), và các đặc trưng kỹ thuật đã được tính toán sẵn (RSI, MACD, dải Bollinger, ATR, phần trăm thay đổi giá). Việc phân vùng thêm theo `year=`/`month=`/`day=` — mịn hơn so với bộ dữ liệu phục vụ — phản ánh đúng mẫu hình truy cập khác biệt của bộ dữ liệu này: nó được đọc theo lô (batch) trong các phiên huấn luyện/tinh chỉnh mô hình định kỳ, quét theo cửa sổ ngày cụ thể, chứ không được đọc trực tiếp bởi các API phục vụ người dùng cuối theo thời gian thực.

**Vì sao chọn định dạng cột (Parquet) thay vì định dạng hàng của RDBMS.** Việc lựa chọn định dạng lưu trữ cột, đọc bằng thư viện xử lý dữ liệu dạng bảng hiệu năng cao (Pandas/Polars) thay vì hàng dữ liệu quan hệ, dựa trên đặc thù mẫu hình truy vấn chi phối của hệ thống là **quét theo dải** (range scan) để tính chỉ báo kỹ thuật hoặc dựng cửa sổ ngữ cảnh (context window) đầu vào cho Chronos-2 — với mẫu hình này, định dạng cột chỉ cần đọc đúng các cột giá trị cần thiết (open/high/low/close/volume) trên toàn bộ dải hàng được yêu cầu, tránh chi phí I/O đọc dư thừa các cột không dùng đến vốn không thể tránh khỏi với định dạng hàng. Kết hợp với cắt tỉa phân vùng ở tầng hệ thống tệp, tổng chi phí I/O cho một truy vấn dựng biểu đồ hoặc dựng ngữ cảnh dự báo được giảm thiểu ở cả hai chiều — chiều cột và chiều phân vùng — điều mà một bảng quan hệ đơn lẻ không phân vùng vật lý theo tài sản/thời gian khó đạt được với cùng mức chi phí vận hành.

**Lược đồ tệp kê khai mô hình (`manifest/latest.json`).** Trong kho lưu trữ mô hình học máy, bên cạnh các artifact trọng số, tồn tại một tệp kê khai tại đường dẫn `manifest/latest.json` với lược đồ tối giản, cốt lõi là một trường duy nhất:

```json
{
  "active_model_s3_uri": "s3://<model-bucket>/models/<phiên-bản>/"
}
```

Tệp kê khai này đóng vai trò một **con trỏ quảng bá phiên bản mô hình** (model promotion pointer): khi dịch vụ suy luận khởi động, nó đọc tệp kê khai này trước tiên; nếu tệp tồn tại và hợp lệ, artifact tại `active_model_s3_uri` được ưu tiên nạp thay cho URI mô hình mặc định cấu hình sẵn; nếu tệp không tồn tại hoặc không hợp lệ, hệ thống quay lại nạp URI mặc định. Cơ chế này cho phép quản trị viên vận hành "quảng bá" (promote) một phiên bản mô hình mới vào môi trường suy luận chỉ bằng cách cập nhật một tệp JSON duy nhất, tách biệt hoàn toàn khỏi chu trình triển khai mã nguồn ứng dụng — tuy nhiên, cần lưu ý rằng vì mô hình được nạp một lần và giữ cố định trong bộ nhớ tiến trình trong suốt vòng đời của nó (mục 2.4.1), sự thay đổi trong tệp kê khai chỉ có hiệu lực từ **lần khởi động tiến trình kế tiếp**, không có hiệu lực tức thời trên một tiến trình đang chạy.

## 3.2. Mô hình xử lý (Process Model)

### 3.2.1. Use case chi tiết

**Bảng 3.1 – Đặc tả Use case UC-01: Dự báo giá bằng AI**

| Trường | Nội dung |
|---|---|
| Mã use case | UC-01 |
| Tên use case | Dự báo giá bằng AI |
| Tác nhân | Khách hàng (End User) |
| Mô tả tóm tắt | Khách hàng yêu cầu hệ thống sinh một dự báo giá xác suất cho tài sản và khung thời gian đang xem, dựa trên dữ liệu nến gần nhất và mô hình Chronos-2. |
| Điều kiện tiên quyết | Khách hàng đã đăng nhập thành công (có token hợp lệ); đã chọn một tài sản và khung thời gian trên giao diện; dữ liệu biểu đồ đã được tải. |
| Luồng sự kiện chính | 1. Khách hàng chọn khung dự báo (horizon) và nhấn nút "Tạo dự đoán".<br>2. Hệ thống lấy lô nến mới nhất từ kho dữ liệu phục vụ (mục 3.1.2) để đảm bảo dữ liệu đầu vào không bị lỗi thời.<br>3. Hệ thống chuẩn hóa và giới hạn số lượng nến đầu vào theo cửa sổ ngữ cảnh phù hợp cho mô hình.<br>4. Hệ thống gửi yêu cầu suy luận đến dịch vụ Chronos-2 kèm ngữ cảnh giá và thông tin cảm xúc thị trường (sentiment).<br>5. Mô hình trả về mảng dự báo đa bước cùng các phân vị xác suất (dải tin cậy, dải biến động).<br>6. Hệ thống đóng gói kết quả thành đối tượng phản hồi có cấu trúc và trả về giao diện.<br>7. Giao diện vẽ chồng dự báo lên biểu đồ nến hiện tại. |
| Luồng ngoại lệ | – Nếu số lượng nến khả dụng nhỏ hơn ngưỡng tối thiểu: hệ thống báo lỗi yêu cầu chờ đồng bộ thêm dữ liệu, không gọi mô hình.<br>– Nếu dịch vụ suy luận phản hồi lỗi hoặc quá thời gian chờ: hệ thống trả về thông báo lỗi tương ứng (mạng, xác thực, hoặc dịch vụ quá tải) mà không làm sập giao diện. |
| Hậu điều kiện | Kết quả dự báo được lưu vào trạng thái phía giao diện, sẵn sàng làm đầu vào cho Use case UC-03 (Phân tích bởi Hội đồng AI, xem mục 3.2.3). |

**Bảng 3.2 – Đặc tả Use case UC-02: Cập nhật dữ liệu thị trường định kỳ**

| Trường | Nội dung |
|---|---|
| Mã use case | UC-02 |
| Tên use case | Cập nhật dữ liệu thị trường định kỳ |
| Tác nhân | Tiến trình nền tự động (Background Ingestion Process) — một tiến trình vòng lặp chạy nội tại bên trong container backend, không phải một dịch vụ lập lịch riêng biệt của nhà cung cấp đám mây |
| Mô tả tóm tắt | Định kỳ mỗi 15 phút, tiến trình nền tự động thu thập dữ liệu thị trường mới nhất từ các nguồn bên ngoài, tính toán đặc trưng kỹ thuật, và ghi bổ sung vào kho dữ liệu làm giàu đặc trưng phục vụ huấn luyện mô hình. |
| Điều kiện tiên quyết | Container backend đang chạy ở chế độ có bật tiến trình thu thập dữ liệu; có kết nối mạng ra ngoài đến API của sàn giao dịch và nguồn dữ liệu vĩ mô. |
| Luồng sự kiện chính | 1. Tiến trình nền được kích hoạt theo chu kỳ thời gian cố định (15 phút/lần) trong suốt vòng đời của container.<br>2. Với từng tài sản trong danh mục theo dõi, tiến trình gọi API công khai của sàn giao dịch để lấy nến giá giao ngay mới nhất.<br>3. Tiến trình gọi thêm API hợp đồng tương lai để lấy tỷ lệ tài trợ, khối lượng mở, và tỷ lệ vị thế mua/bán.<br>4. Tiến trình truy vấn nguồn dữ liệu vĩ mô bên ngoài để lấy chỉ số USD và lợi suất trái phiếu kho bạc kỳ hạn 10 năm.<br>5. Tiến trình tính toán các đặc trưng kỹ thuật phái sinh (RSI, MACD, dải Bollinger, ATR, phần trăm thay đổi) trên dữ liệu vừa thu thập.<br>6. Tiến trình ghi bổ sung dữ liệu đã làm giàu vào kho lưu trữ, theo đúng cấu trúc phân vùng `symbol=/year=/month=/day=` của ngày hiện tại.<br>7. Tiến trình ghi log kết quả thực thi (số tài sản xử lý thành công/thất bại) rồi tạm nghỉ cho đến chu kỳ kế tiếp. |
| Luồng ngoại lệ | – Nếu một nguồn dữ liệu bên ngoài (sàn giao dịch hoặc dữ liệu vĩ mô) tạm thời không phản hồi: tiến trình bỏ qua tài sản/nguồn đó trong chu kỳ hiện tại, ghi log cảnh báo, và tiếp tục xử lý các tài sản còn lại — một lần thu thập thất bại không làm dừng toàn bộ tiến trình nền. |
| Hậu điều kiện | Kho dữ liệu làm giàu đặc trưng được cập nhật với dữ liệu mới nhất, sẵn sàng cho chu kỳ huấn luyện/tinh chỉnh mô hình kế tiếp (mục 2.4.1). |

### 3.2.2. Sơ đồ tuần tự — Quy trình xử lý yêu cầu Dự báo giá

Luồng xử lý một yêu cầu dự báo giá (tương ứng Use case UC-01) đi qua các thành phần theo trình tự sau:

1. **Client (ứng dụng React)** gửi yêu cầu HTTPS đến tên miền công khai của hệ thống, kèm token xác thực trong tiêu đề `Authorization: Bearer <token>`.
2. **CloudFront** tiếp nhận yêu cầu tại điểm hiện diện biên gần nhất, chuyển tiếp đến origin là máy chủ backend; nhóm bảo mật (security group) của máy chủ backend chỉ chấp nhận lưu lượng có nguồn gốc từ danh sách tiền tố quản lý (prefix list) của CloudFront, không chấp nhận kết nối trực tiếp từ Internet.
3. **Caddy** (chạy trên máy chủ EC2 backend) tiếp nhận kết nối đã được giải mã TLS, thực hiện vai trò reverse proxy chuyển tiếp yêu cầu đến tiến trình FastAPI cục bộ (Uvicorn, cổng nội bộ 8000).
4. **FastAPI** tiếp nhận yêu cầu tại route `/predict`, kiểm tra lược đồ dữ liệu đầu vào (Pydantic) trước khi xử lý.
5. **Dependency xác thực Cognito** được gọi trước khi vào logic nghiệp vụ: lấy bộ khóa công khai (JWKS) từ Cognito, xác minh chữ ký RS256 của token, kiểm tra bên phát hành (issuer), và kiểm tra thủ công trường audience/client tương ứng với loại token (`id` hoặc `access`). Nếu xác thực thất bại, yêu cầu bị từ chối ngay tại đây, không đi tiếp đến các bước sau.
6. **Truy xuất dữ liệu nến** từ kho S3 Parquet (mục 3.1.2) theo phân vùng `symbol=`/`timeframe=` tương ứng làm nền lịch sử, đồng thời bổ sung các nến mới nhất trực tiếp từ Binance nếu dữ liệu trên S3 chưa đủ mới — đảm bảo cửa sổ ngữ cảnh đưa vào mô hình luôn phản ánh trạng thái giá gần nhất ngay cả khi tiến trình cập nhật kho dữ liệu định kỳ có độ trễ.
7. **Tính toán/tổng hợp dữ liệu cảm xúc thị trường (sentiment)** — bao gồm một thành phần tính toán nội tại từ chính hành vi giá gần nhất, và một thành phần bên ngoài tổng hợp từ nguồn tin tức/mạng xã hội/vĩ mô.
8. **Dịch vụ suy luận (Model Loader + Inference Service)** giải quyết mô hình đang hoạt động theo tệp kê khai `manifest/latest.json` (mục 3.1.2) — nếu mô hình đã được nạp sẵn từ lần khởi động tiến trình, bước này chỉ tái sử dụng mô hình đã có trong bộ nhớ (không tải lại).
9. **Chronos-2** thực hiện suy luận zero-shot trên cửa sổ ngữ cảnh vừa dựng, trả về mảng dự báo đa bước cùng các phân vị xác suất tương ứng.
10. **FastAPI** đóng gói kết quả thành đối tượng phản hồi JSON có cấu trúc (dự báo, dải tin cậy, dải biến động, điểm cảm xúc) và trả về qua đúng chuỗi thành phần đã đi qua (Caddy → CloudFront → Client).
11. **Client** nhận phản hồi, cập nhật trạng thái, và vẽ lớp phủ dự báo lên biểu đồ nến đang hiển thị.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ TUẦN TỰ QUY TRÌNH DỰ BÁO GIÁ TẠI ĐÂY]`

### 3.2.3. Sơ đồ hoạt động — Hội đồng cố vấn AI (AI Council)

Sau khi Use case UC-01 hoàn tất và có kết quả dự báo từ Chronos-2, người dùng có thể kích hoạt thêm một luồng xử lý riêng biệt — Hội đồng cố vấn AI — thực thi tuần tự qua ba hoạt động, mỗi hoạt động là một tác nhân CrewAI (mục 2.2.3):

1. **Nhà phân tích định lượng (Quant Analyst)** tiếp nhận mảng dự báo, dải tin cậy/biến động từ Chronos-2 cùng dữ liệu cảm xúc thị trường, thực hiện suy luận ngôn ngữ để hình thành một nhận định xu hướng có lập luận đi kèm — hoạt động này là điểm bắt đầu của luồng, không có điều kiện rẽ nhánh.
2. **Nhà quản trị rủi ro (Risk Manager)** tiếp nhận đầu ra của hoạt động trước, tính toán và giới hạn các thông số quản trị rủi ro trong biên độ an toàn đã định trước (biên độ đòn bẩy tối đa, tỷ lệ kích thước vị thế tối đa theo khẩu vị rủi ro mặc định của hệ thống), xác định vùng vào lệnh, điểm dừng lỗ, và hai mốc chốt lời (chốt lời an toàn gần và chốt lời mục tiêu xa).
3. **Thẩm phán thực thi (Execution Judge)** tiếp nhận đầu ra của cả hai hoạt động trước, kiểm tra tính nhất quán logic (ví dụ: hành động đề xuất phải phù hợp chiều với nhận định xu hướng; các mốc giá phải theo đúng thứ tự hợp lý tương ứng với hành động LONG/SHORT), rồi phát ra quyết định cuối cùng dưới dạng đối tượng có cấu trúc (hành động, độ tin cậy, toàn bộ thông số rủi ro, lập luận tổng hợp).

Vì đây là một tiến trình tuần tự đơn hướng (không có nhánh điều kiện quay lại một hoạt động trước đó), toàn bộ luồng được truyền trực tiếp (streaming) từng dòng nhật ký suy luận của các tác nhân về giao diện người dùng ngay khi được sinh ra, thông qua kênh Server-Sent Events — cho phép người dùng quan sát diễn biến lập luận theo thời gian thực thay vì chỉ nhận kết quả cuối cùng sau một khoảng chờ không rõ tiến trình.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ HOẠT ĐỘNG AI COUNCIL (QUANT ANALYST → RISK MANAGER → EXECUTION JUDGE) TẠI ĐÂY]`

## 3.3. Hệ thống màn hình và giao diện (UI Design)

Giao diện được thiết kế theo định hướng một **đầu cuối giao dịch chuyên nghiệp** (institutional-grade trading terminal), lấy cảm hứng từ các nền tảng giao dịch định chế, với ba nguyên tắc thiết kế chủ đạo:

**Bố cục ba cột (Three-column Layout).** Màn hình chính được chia thành ba cột chức năng độc lập nhưng đồng bộ trạng thái: cột trái là danh sách tài sản có thể tìm kiếm/lọc; cột giữa (chiếm phần lớn không gian còn lại) là biểu đồ nến chính cùng các chỉ báo kỹ thuật; cột phải là bảng điều khiển dự báo và Hội đồng cố vấn AI. Ba cột được canh chỉnh để tự nhiên cân bằng chiều cao theo cột có nội dung dài nhất, đồng thời mỗi cột bên (trái/phải) có khả năng cuộn nội bộ độc lập, không phụ thuộc vào chiều cao của cột biểu đồ ở giữa — đảm bảo người dùng luôn nhìn thấy biểu đồ chính đầy đủ trong khi vẫn có thể cuộn xem chi tiết bảng điều khiển dự báo hoặc lịch sử suy luận của Hội đồng AI.

**Giao diện tối (Dark Mode).** Toàn bộ giao diện sử dụng bảng màu tối (nền tím than sẫm, các khối nội dung dạng "kính mờ" — glass panel — với hiệu ứng làm mờ nền và điểm nhấn màu cyan/tím neon) — một lựa chọn thiết kế có chủ đích cho một công cụ mà người dùng thường theo dõi liên tục trong thời gian dài, giúp giảm mỏi mắt so với giao diện nền sáng, đồng thời tăng độ tương phản cho các yếu tố dữ liệu quan trọng (nến tăng/giảm, chỉ báo, nhãn cảnh báo rủi ro).

**Thư viện biểu đồ Lightweight Charts.** Thành phần hiển thị biểu đồ được xây dựng trên thư viện `lightweight-charts`, một lựa chọn có chủ đích để xử lý khối lượng dữ liệu nến lớn (hàng nghìn nến lịch sử cộng dồn với luồng cập nhật thời gian thực liên tục) mà không gây giật/lag trên trình duyệt — thư viện này render trực tiếp trên canvas thay vì cây DOM (vốn không thể mở rộng tuyến tính với số lượng phần tử lớn), và cung cấp API cập nhật gia tăng (incremental update) cho phép thêm/sửa một nến duy nhất khi có dữ liệu thời gian thực mới mà không cần vẽ lại toàn bộ biểu đồ. Trên cùng một khung nhìn biểu đồ, hệ thống vẽ chồng thêm một chuỗi nến dự báo (với màu sắc phân biệt) cùng dải tin cậy/biến động ngay khi có kết quả từ Use case UC-01, và hỗ trợ thao tác nhấp chuột để ghim một bảng thông tin OHLC chi tiết tại bất kỳ điểm nào trên biểu đồ (đóng bằng phím Esc). Bên dưới biểu đồ chính là hai biểu đồ phụ đồng bộ trục thời gian, hiển thị chỉ báo RSI(14) và MACD(12,26,9).

**Bảng điều khiển quản trị rủi ro.** Khi Hội đồng cố vấn AI hoàn tất phân tích (mục 3.2.3), cột phải hiển thị một bộ thẻ thông tin có cấu trúc: huy hiệu hành động (LONG/SHORT/HOLD) kèm phần trăm độ tin cậy; khối "Vùng vào lệnh & Thời điểm"; hai thẻ chốt lời phân biệt trực quan (chốt lời an toàn gần và chốt lời mục tiêu xa); một khối hợp nhất điểm dừng lỗ cứng cùng tỷ lệ rủi ro/lợi nhuận tương ứng; điểm vô hiệu hóa khuyến nghị (mức giá mà nếu bị phá vỡ, khuyến nghị không còn giá trị tham khảo); và một khối lập luận tổng hợp bằng văn bản tự nhiên. Toàn bộ nhật ký suy luận chi tiết của từng tác nhân được lưu lại dưới dạng một bảng ghi (transcript) có thể mở rộng/thu gọn, trình bày theo phong cách cửa sổ dòng lệnh (terminal) để nhấn mạnh tính chất "quan sát được" (observable) của quá trình suy luận AI thay vì một hộp đen.

`[CHÚ THÍCH: CHÈN ẢNH GIAO DIỆN TRADING TERMINAL (BỐ CỤC BA CỘT, BIỂU ĐỒ NẾN + RSI/MACD, BẢNG ĐIỀU KHIỂN HỘI ĐỒNG AI) TẠI ĐÂY]`

## 3.4. Hệ thống giám sát và báo biểu (Monitoring & Logging)

**Giám sát qua CloudWatch.** Hạ tầng giám sát được định nghĩa dưới dạng mã nguồn cùng với toàn bộ hạ tầng còn lại (mục 2.3.3), bao gồm một CloudWatch Dashboard tổng hợp các chỉ số vận hành của hệ thống. Ngoài các chỉ số hạ tầng mặc định do AWS cung cấp (CPU, bộ nhớ, mạng của instance EC2), backend chủ động phát ra các **chỉ số tùy chỉnh** (custom metrics) phản ánh trực tiếp sức khỏe nghiệp vụ của API, thông qua một middleware bao quanh mọi yêu cầu HTTP, dưới namespace `AetherForecast/API`:

- **`ApiRequests`** (đơn vị: Count) — tăng 1 cho mỗi yêu cầu API được xử lý, bất kể kết quả thành công hay lỗi.
- **`ApiLatencyMs`** (đơn vị: Milliseconds) — thời gian xử lý thực tế của mỗi yêu cầu, đo từ lúc bắt đầu đến lúc trả phản hồi.
- **`Api5xx`** (đơn vị: Count) — chỉ được phát ra khi mã trạng thái phản hồi từ 500 trở lên, cho phép tách riêng tỷ lệ lỗi phía máy chủ khỏi tổng lưu lượng.

Cả ba chỉ số đều được gắn kèm chiều dữ liệu (dimension) `Service=backend`, cho phép truy vấn/lọc theo dịch vụ khi hệ thống mở rộng thêm nhiều thành phần trong tương lai. Việc phát chỉ số được thiết kế theo nguyên tắc **không chặn luồng xử lý chính** (best-effort, non-blocking): nếu lệnh gọi CloudWatch thất bại (do sự cố mạng tạm thời hoặc giới hạn tốc độ gọi API), lỗi chỉ được ghi log cảnh báo mà không làm gián đoạn hay trả lỗi cho yêu cầu của người dùng đang được xử lý — đảm bảo tầng giám sát không bao giờ trở thành điểm lỗi đơn (single point of failure) ảnh hưởng đến tầng nghiệp vụ.

**Ghi log có cấu trúc bằng `structlog`.** Toàn bộ log của backend được cấu hình thông qua thư viện `structlog`, thay thế cho log dạng văn bản tự do (plain-text) mặc định của thư viện `logging` chuẩn trong Python. Cấu hình ghi log tổng hợp nhiều bộ xử lý (processor) theo thứ tự: gộp các biến ngữ cảnh (context variables) đã được thiết lập trong quá trình xử lý yêu cầu, gắn thêm cấp độ log và tên logger, gắn dấu thời gian theo chuẩn ISO-8601 (múi giờ UTC), và định dạng thông tin ngoại lệ/stack trace nếu có — toàn bộ được kết xuất cuối cùng dưới dạng một **đối tượng JSON** trên mỗi dòng log, thay vì một chuỗi văn bản không có cấu trúc. Lợi ích thực tiễn của cách tiếp cận này là log có thể được truy vấn/lọc theo trường dữ liệu cụ thể (ví dụ: lọc theo cấp độ lỗi, theo tên endpoint, hoặc theo bất kỳ trường ngữ cảnh nào được đính kèm) bằng các công cụ tổng hợp log tiêu chuẩn, thay vì phải phân tích cú pháp (parse) chuỗi văn bản tự do bằng biểu thức chính quy — một yêu cầu thiết yếu để phục vụ cả mục đích gỡ lỗi vận hành (operational debugging) lẫn tái dựng dấu vết kiểm toán (audit trail reconstruction) khi cần điều tra một sự cố hoặc một quyết định cụ thể của Hội đồng cố vấn AI sau khi sự việc đã xảy ra.

---

# CHƯƠNG 4: THỬ NGHIỆM VÀ ĐÁNH GIÁ

## 4.1. Các kịch bản thử nghiệm

Đề tài thiết kế ba kịch bản thử nghiệm độc lập, mỗi kịch bản nhắm vào một thuộc tính chất lượng khác nhau của hệ thống đã trình bày ở Chương 3:

**Kịch bản 1 — Hiệu năng hệ thống (System Performance).** Đo lường và so sánh độ trễ (latency) giữa hai đường xử lý dữ liệu có bản chất khác biệt căn bản đã được làm rõ ở mục 3.1.2: API `/chart` (proxy đọc trực tiếp từ Binance, không suy luận) và API `/predict` (đọc kết hợp S3/Binance, có suy luận mô hình học sâu). Mục tiêu là định lượng chênh lệch độ trễ giữa một đường xử lý thuần I/O và một đường xử lý có xen kẽ tính toán tensor nặng, làm cơ sở đánh giá trải nghiệm người dùng thực tế.

**Kịch bản 2 — Độ chính xác mô hình (Model Accuracy).** Vì Chronos-2 là một mô hình dự báo xác suất (mục 2.2.1), tiêu chí đánh giá phù hợp không phải là sai số giữa một dự báo điểm và giá thực tế (vốn chỉ có ý nghĩa với mô hình hồi quy điểm truyền thống), mà là **độ phủ của dải tin cậy** (confidence band coverage) — tỷ lệ phần trăm các trường hợp mà giá trị giá thực tế quan sát được về sau rơi vào đúng bên trong dải phân vị đã dự báo.

**Kịch bản 3 — Rà soát bảo mật và khả năng phục hồi theo mô hình đe dọa (Threat-model-guided Security & Resilience Review).** Đây là một hoạt động rà soát mã nguồn thủ công có định hướng theo mô hình đe dọa (threat model) đã trình bày ở mục 2.2.4 và 1.2.3 — đối chiếu từng lớp phòng thủ đã được thiết kế với hành vi thực tế của mã nguồn đang triển khai, thay vì một bài kiểm tra thâm nhập tự động bằng công cụ thương mại của bên thứ ba. Phạm vi rà soát bao gồm: kiểm tra tính hợp lệ đầu vào tại các điểm dựng đường dẫn lưu trữ đối tượng từ tham số người dùng, đối chiếu cấu hình mạng của các máy chủ EC2 với chủ trương "không SSH" đã đặt ra, và đánh giá rủi ro của việc nạp mã nguồn từ mô hình bên thứ ba (`trust_remote_code=True`, mục 2.2.1).

## 4.2. Kết quả thử nghiệm các kịch bản

### Hiệu năng

Về mặt kiến trúc, hai API được kỳ vọng có đặc tính độ trễ khác biệt rõ rệt, dựa trên bản chất xử lý đã phân tích ở mục 3.1.2 và 3.2.2:

- API **`/chart`** chỉ thực hiện một lệnh gọi HTTP đơn thuần đến API công khai của Binance và ánh xạ lại định dạng phản hồi — không có bất kỳ phép biến đổi tensor hay suy luận mô hình nào — nên độ trễ của API này về bản chất bị chi phối hoàn toàn bởi độ trễ mạng đến Binance, được kỳ vọng ở mức thấp và ổn định.
- API **`/predict`** phải thực hiện tuần tự: truy xuất/hợp nhất dữ liệu (S3 và/hoặc Binance), dựng cửa sổ ngữ cảnh, và quan trọng nhất là **suy luận tensor trên mô hình nền tảng Chronos-2** — một phép toán có chi phí tính toán lớn hơn nhiều bậc so với một lệnh gọi I/O đơn thuần, đặc biệt khi môi trường suy luận hiện tại không có phân bổ GPU chuyên dụng như pipeline huấn luyện trên AWS Batch (mục 2.3.3) mà chạy trên tài nguyên tính toán chung của máy chủ backend — nên độ trễ của API này được kỳ vọng cao hơn đáng kể và có phương sai lớn hơn, phụ thuộc vào độ dài cửa sổ ngữ cảnh và số bước dự báo (horizon) được yêu cầu.

Một chi tiết thiết kế quan trọng làm giảm nhẹ tác động của độ trễ suy luận đến trải nghiệm tổng thể của hệ thống: route xử lý `/predict` được định nghĩa dưới dạng hàm đồng bộ (`def`, không phải `async def`) trong FastAPI — theo cơ chế của framework, một route đồng bộ được tự động thực thi trong một threadpool riêng biệt với vòng lặp sự kiện (event loop) chính, nghĩa là trong lúc một yêu cầu `/predict` đang thực hiện suy luận tensor (có thể kéo dài), vòng lặp sự kiện chính vẫn hoàn toàn rảnh để tiếp tục phục vụ đồng thời các kết nối WebSocket dữ liệu thời gian thực và các luồng Server-Sent Events của Hội đồng AI cho những người dùng khác — tách biệt hoàn toàn giữa tác vụ nặng CPU/tensor và các tác vụ I/O-bound thời gian thực chi phối phần còn lại của hệ thống.

*Về số liệu đo đạc cụ thể:* đề tài xác lập phương pháp đo (lặp lại N lần trong điều kiện mạng ổn định, ghi nhận trung vị và phân vị 95 — p95 — cho từng API, phân biệt rõ giữa lượt gọi đầu — cold — và các lượt gọi sau khi đã có dữ liệu đệm) nhưng **số liệu định lượng thực nghiệm cụ thể cần được đo đạc trực tiếp trên môi trường đã triển khai** (không phải môi trường phát triển cục bộ) để phản ánh đúng điều kiện mạng và tải thực tế, và sẽ được bổ sung vào vị trí biểu đồ dưới đây sau khi hoàn tất đo đạc.

`[CHÚ THÍCH: CHÈN BIỂU ĐỒ ĐỘ TRỄ API (/chart SO VỚI /predict, TRUNG VỊ VÀ P95) DỰA TRÊN SỐ LIỆU ĐO THỰC TẾ TRÊN MÔI TRƯỜNG TRIỂN KHAI TẠI ĐÂY]`

### Độ chính xác AI

Phương pháp đánh giá được áp dụng là **kiểm định độ phủ dải tin cậy theo cửa sổ trượt** (rolling-window band coverage validation): với mỗi điểm thời gian trong tập dữ liệu lịch sử giữ lại để đánh giá (không sử dụng trong bất kỳ bước huấn luyện/tinh chỉnh nào), hệ thống dựng dự báo Chronos-2 dựa trên dữ liệu quá khứ tính đến điểm đó, rồi đối chiếu giá trị thực tế đã xảy ra sau đó với dải phân vị dưới/trên tương ứng đã dự báo. Cần lưu ý rằng cấu hình phân vị mặc định của hệ thống (mục 3.1, lược đồ `PredictRequest`) là `[0.1, 0.5, 0.9]` — tức dải tin cậy mặc định là **dải 80% danh nghĩa** (P10–P90), không phải P5–P95; hệ thống cho phép cấu hình lại tập phân vị theo yêu cầu (ví dụ `[0.05, 0.5, 0.95]` cho dải 90%) nên việc đánh giá cần thực hiện tại đúng mức phân vị được sử dụng trong thực tế triển khai. Tiêu chí đạt yêu cầu là **tỷ lệ phủ thực nghiệm** (thực tế rơi vào trong dải) phải xấp xỉ đúng mức danh nghĩa của dải đã cấu hình (ví dụ ~80% cho dải P10–P90) trên cả hai lớp tài sản đại diện đã xác lập ở mục 1.3.1 (Binance Spot biến động cao và XAUUSD biến động thấp hơn) — một mô hình có tỷ lệ phủ thực nghiệm lệch quá xa so với mức danh nghĩa (quá cao — dải quá rộng, kém hữu ích; hoặc quá thấp — dải quá hẹp, đánh giá thấp rủi ro thực tế) được xem là hiệu chuẩn kém (poorly calibrated).

*Về số liệu định lượng cụ thể:* tỷ lệ phủ thực nghiệm chính xác trên từng lớp tài sản cần được tính toán từ một lượt chạy backtest đầy đủ trên dữ liệu lịch sử giữ lại, và sẽ được trình bày dưới dạng biểu đồ so sánh tỷ lệ phủ thực nghiệm với mức danh nghĩa theo từng mức phân vị.

`[CHÚ THÍCH: CHÈN BIỂU ĐỒ ĐÁNH GIÁ ĐỘ PHỦ CHRONOS-2 (TỶ LỆ PHỦ THỰC NGHIỆM SO VỚI MỨC DANH NGHĨA, THEO TỪNG LỚP TÀI SẢN) TẠI ĐÂY]`

### Bảo mật

Rà soát bảo mật theo Kịch bản 3 ghi nhận ba phát hiện chính, được trình bày theo đúng tinh thần minh bạch đã xác lập ở mục 2.2.4 — phân định rõ giữa phát hiện đã được khắc phục ngay trong phạm vi đề tài và phát hiện còn tồn đọng (được tổng hợp lại đầy đủ ở mục 5.2):

1. **Thiếu kiểm tra hợp lệ theo danh sách trắng ký tự (whitelist) cho tham số `symbol`** tại hai API `/chart` và `/predict` — trước rà soát, tham số này chỉ được ràng buộc độ dài (2–20 ký tự) mà không ràng buộc tập ký tự hợp lệ, khác với API WebSocket thời gian thực (`/ws/{symbol}`) vốn đã có sẵn một biểu thức chính quy whitelist (`^[A-Z0-9]{2,20}$`). Vì tham số này được dùng để dựng một phần đường dẫn truy vấn đối tượng lưu trữ (`symbol=<SYMBOL>/`, mục 3.1.2), việc thiếu ràng buộc ký tự tạo ra một bề mặt tấn công dạng **chèn mẫu đường dẫn** (path/glob pattern injection) — một giá trị `symbol` được soạn có chủ đích (chứa ký tự đặc biệt hoặc chuỗi duyệt thư mục) có thể khiến hệ thống dựng một đường dẫn truy vấn lưu trữ nằm ngoài phạm vi dự kiến, tiềm ẩn rủi ro từ chối dịch vụ (DoS) do quét một khối lượng đối tượng bất thường hoặc lỗi xử lý ngoài dự kiến. **Phát hiện này đã được khắc phục ngay trong phạm vi đề tài** (mục 4.3) bằng cách bổ sung một hàm kiểm tra hợp lệ dùng chung, áp dụng cùng biểu thức chính quy đã có sẵn trên route WebSocket, cho cả hai API `/chart` và `/predict`.
2. **Rủi ro thực thi mã từ xa qua chuỗi cung ứng mô hình (`trust_remote_code=True`)** — như đã phân tích ở mục 2.2.1 và 2.2.4, việc nạp mô hình Chronos-2 sử dụng tùy chọn `trust_remote_code=True` tại nhiều điểm trong `model_loader.py`, cho phép thực thi mã Python đi kèm trong artifact mô hình mà không có một cơ chế xác minh chữ ký/mã băm theo danh sách cho phép trước khi thực thi — cơ chế mã băm hiện có (mục 2.4.1) chỉ phục vụ mục đích xác định cache đã lỗi thời hay chưa. Đây là một rủi ro **còn tồn đọng**, được đưa vào định hướng phát triển ở mục 5.3.
3. **Cấu hình truy cập từ xa không đồng nhất giữa hai máy chủ EC2** — máy chủ backend (production) đã tuân thủ đúng chủ trương "không SSH" (mục 2.2.4, xác nhận qua cấu hình nhóm bảo mật chỉ mở cổng 443 từ CloudFront); tuy nhiên máy chủ EC2 dành cho huấn luyện lại có cổng 22 được mở với dải địa chỉ nguồn mặc định là toàn bộ Internet (`0.0.0.0/0`) trong định nghĩa hạ tầng dưới dạng mã nguồn, chỉ được thu hẹp nếu người vận hành chủ động ghi đè tham số khi triển khai. Đây là một sự **không nhất quán trong áp dụng chính sách bảo mật** giữa hai máy chủ có cùng chủ trương thiết kế, được đưa vào danh sách tồn đọng ở mục 5.2.

`[CHÚ THÍCH: CHÈN BÁO CÁO RÀ SOÁT BẢO MẬT / DANH SÁCH PHÁT HIỆN (VULNERABILITY FINDINGS) CHI TIẾT TẠI ĐÂY]`

## 4.3. Xử lý các trường hợp ngoại lệ

**Ứng xử của hệ thống trước biến động thị trường cực đoan.** Kiến trúc tuần tự của Hội đồng cố vấn AI (mục 2.2.3, 3.2.3) không chỉ là một lựa chọn về hiệu năng mà còn đóng vai trò một cơ chế phòng vệ khi thị trường biến động bất thường: tác nhân Nhà quản trị rủi ro được thiết kế với các quy tắc thận trọng tường minh — điểm cảm xúc thị trường ở mức cực đoan (lớn hơn 0,7 hoặc nhỏ hơn -0,7), chỉ số Sợ hãi/Tham lam ở vùng cực đoan (dưới 20 hoặc trên 80), hoặc biến động thực hiện (realized volatility) ở mức cao đều là tín hiệu buộc tác nhân này phải hạ thấp đòn bẩy/kích thước vị thế đề xuất, hoặc trong tình huống nghiêm trọng, **phủ quyết** hoàn toàn giao dịch. Khi Nhà quản trị rủi ro phủ quyết, tác nhân Thẩm phán thực thi bị ràng buộc chặt (tại tầng chỉ dẫn của tác nhân) phải trả về hành động **HOLD** với toàn bộ thông số giao dịch (vùng vào lệnh, dừng lỗ, chốt lời) bằng 0 — đảm bảo hệ thống không bao giờ đưa ra một khuyến nghị giao dịch cụ thể trong điều kiện thị trường mà chính các tác nhân đánh giá là quá bất định để đưa ra quyết định có trách nhiệm.

**Các biện pháp khắc phục bảo mật đã áp dụng ngay trong phạm vi đề tài.** Ứng với hai trong ba phát hiện ở mục 4.2:

- **Whitelist biểu thức chính quy cho tham số `symbol`:** một hàm kiểm tra hợp lệ dùng chung (`normalize_and_validate_symbol`) được bổ sung vào lược đồ dữ liệu dùng chung của mô-đun ML (`src/ml/schemas.py`), tái sử dụng đúng biểu thức chính quy `^[A-Z0-9]{2,20}$` đã có sẵn trên route WebSocket thời gian thực — chuẩn hóa (chuyển hoa, loại khoảng trắng thừa) trước khi kiểm tra để không phá vỡ khả năng tương thích với giá trị chữ thường hợp lệ, và từ chối ngay lập tức (phản hồi 422) mọi giá trị chứa ký tự nằm ngoài tập chữ cái/chữ số. Hàm này được áp dụng đồng thời cho lược đồ `PredictRequest` (thông qua trình xác thực trường dữ liệu của Pydantic) và cho tham số đường dẫn của API `/chart/{symbol}` — đóng lại khoảng trống nhất quán giữa ba API cùng nhận tham số `symbol` từ người dùng.
- **Không sử dụng SSH cho máy chủ production, thay bằng AWS Systems Manager Session Manager:** biện pháp này thực chất đã được áp dụng từ trước (không phải một khắc phục mới trong lượt rà soát này) cho riêng máy chủ backend phục vụ người dùng cuối — nhóm bảo mật của máy chủ này không mở cổng 22 dưới bất kỳ điều kiện nào, và toàn bộ thao tác triển khai/quản trị được thực hiện qua kênh quản lý phiên của AWS. Biện pháp này **chưa được áp dụng** cho máy chủ huấn luyện (mục 4.2, phát hiện 3) — việc thu hẹp dải địa chỉ nguồn mặc định cho máy chủ này đòi hỏi một quyết định thay đổi tham số hạ tầng, được ghi nhận là hạng mục tồn đọng ưu tiên ở mục 5.2 thay vì được tự ý thay đổi ngoài phạm vi yêu cầu của đợt rà soát này.

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. Kết quả đối chiếu với mục tiêu

**Bảng 5.1 – Đối chiếu tiêu chí chức năng (Bảng 1.1) với kết quả đạt được**

| STT | Tiêu chí chức năng | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Dữ liệu thị trường thời gian thực | Đạt | Kênh WebSocket thời gian thực và API `/chart` proxy trực tiếp Binance đã vận hành (mục 3.1.2, 3.2.2), phục vụ hiển thị biểu đồ và chỉ báo kỹ thuật liên tục. |
| 2 | Dự báo giá bằng Foundation Model | Đạt | Chronos-2 tích hợp qua API `/predict`, sinh dự báo xác suất đa bước cùng dải tin cậy/biến động theo cơ chế zero-shot (mục 2.2.1, 3.2.2). |
| 3 | Hội đồng cố vấn AI (AI Council) | Đạt | Ba tác nhân CrewAI tuần tự (Quant Analyst → Risk Manager → Execution Judge) tổng hợp khuyến nghị có cấu trúc, truyền trực tiếp qua SSE (mục 2.2.3, 3.2.3). |
| 4 | Đa ngôn ngữ | Đạt | Giao diện song ngữ Việt/Anh với phát hiện và ghi nhớ lựa chọn ngôn ngữ giữa các phiên. |
| 5 | Xác thực người dùng | Đạt | Đăng ký/đăng nhập qua Cognito, xác thực token ở mọi API nghiệp vụ (mục 2.3.3, 3.2.2). |

**Bảng 5.2 – Đối chiếu tiêu chí phi chức năng (Bảng 1.2) với kết quả đạt được**

| STT | Tiêu chí phi chức năng | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Khả năng vận hành liên tục khi cập nhật | Đạt một phần | Quy trình canary + tự động rollback đã vận hành (mục 2.2.4); vẫn còn gián đoạn ngắn tại thời điểm chuyển đổi container, chưa đạt mức "không gián đoạn" tuyệt đối (xem mục 5.2, 5.3). |
| 2 | Kiểm soát truy cập hạ tầng | Đạt một phần | Máy chủ production không SSH (SSM-only) đã xác nhận; máy chủ huấn luyện chưa đồng nhất chính sách (mục 4.2, 4.3). |
| 3 | Bảo mật lưu trữ dữ liệu | Đạt | Mã hóa tại chỗ (SSE-S3) và quản lý phiên bản (versioning) đã cấu hình cho cả ba miền lưu trữ (mục 3.1.1). |
| 4 | Kiểm soát truy cập API | Đạt | Xác thực Cognito bắt buộc trên mọi API nghiệp vụ; whitelist ký tự cho tham số `symbol` đã được bổ sung đồng nhất trong Chương 4. |
| 5 | Hiệu quả chi phí hạ tầng tính toán | Đạt | AWS Batch cấp phát GPU đàn hồi (spot/on-demand) cho huấn luyện, không duy trì tài nguyên GPU thường trực (mục 2.3.3). |

## 5.2. Các vấn đề còn tồn đọng

Đề tài ghi nhận minh bạch các hạn chế sau, làm cơ sở cho định hướng phát triển ở mục 5.3:

1. **Độ trễ suy luận cao và chưa có phân bổ tính toán chuyên dụng cho tầng suy luận.** Không giống pipeline huấn luyện (được cấp GPU đàn hồi qua AWS Batch, mục 2.3.3), tầng suy luận phục vụ API `/predict` hiện chạy trên tài nguyên tính toán chung của máy chủ backend, không có phân bổ GPU riêng — khiến chi phí tính toán tensor của Chronos-2 (mục 4.2) trở thành yếu tố giới hạn hiệu năng rõ rệt nhất của toàn hệ thống.
2. **Kiến trúc tác nhân tuần tự vẫn tiềm ẩn thiên kiến xác nhận (confirmation bias).** Vì không có bước phản biện độc lập giữa các tác nhân (mục 2.2.3), mỗi tác nhân trong Hội đồng AI chỉ tiếp tục xây dựng lập luận trên kết luận của tác nhân trước mà không có cơ chế nào chủ động thách thức lại giả định ban đầu — một nhận định lệch của Nhà phân tích định lượng ở bước đầu có thể được truyền nguyên vẹn qua toàn bộ chuỗi mà không bị chất vấn.
3. **Thiếu các cổng kiểm soát DevSecOps tự động hóa nâng cao.** Như đã trình bày minh bạch ở mục 2.2.4 và tổng hợp lại ở mục 4.2: chưa có tường lửa ứng dụng web (WAF) ở lớp biên CDN, chưa bật cơ chế bất biến hóa đối tượng lưu trữ (S3 Object Lock), và chưa có cơ chế xác minh toàn vẹn mô hình học máy bằng chữ ký số trước khi cho phép `trust_remote_code=True` thực thi.
4. **Tiến trình cập nhật định kỳ cho bộ dữ liệu phục vụ thô (`market/klines/`) tồn tại dưới dạng mã nguồn nhưng chưa được triển khai vận hành thực tế** — chỉ được tham chiếu trong ngăn xếp hạ tầng lập lịch hiện chưa được khởi tạo trong ứng dụng CDK gốc (khác với tiến trình cập nhật bộ dữ liệu làm giàu đặc trưng cho huấn luyện, vốn đã vận hành thực tế qua tiến trình nền nội tại, mục 3.2.1). Trong điều kiện vận hành hiện tại, API `/predict` bù đắp khoảng trống này bằng cơ chế bổ sung nến mới nhất trực tiếp từ Binance (mục 3.1.2) — một biện pháp giảm nhẹ hiệu quả nhưng không thay thế hoàn toàn giá trị của một kho dữ liệu lịch sử được làm mới đều đặn và độc lập với tình trạng khả dụng tức thời của API Binance.
5. **Cấu hình truy cập từ xa chưa đồng nhất giữa các máy chủ EC2** (mục 4.2, phát hiện 3) — máy chủ huấn luyện vẫn mở cổng SSH ra toàn bộ Internet theo giá trị mặc định.

## 5.3. Đề xuất kiến trúc nâng cao (Hướng phát triển tương lai)

Mục này trình bày các đề xuất kiến trúc được khảo sát về mặt lý thuyết trong quá trình thực hiện đề tài (đã giới thiệu sơ lược ở mục 2.2.3, 2.2.4) nhưng được xác định nằm ngoài phạm vi hiện thực hóa — đây là định hướng phát triển cho giai đoạn tiếp theo của hệ thống, **không phải mô tả một trạng thái đã triển khai**.

### 5.3.1. Kiến trúc đa tác nhân dạng đồ thị và tác nhân phản biện (Devil's Advocate)

Đề xuất nâng cấp Hội đồng cố vấn AI từ kiến trúc tuần tự CrewAI hiện tại (mục 2.2.3) lên kiến trúc đồ thị trạng thái theo mô hình **LangGraph**, nhằm khắc phục trực tiếp hạn chế về thiên kiến xác nhận đã nêu ở mục 5.2. Thiết kế đề xuất bổ sung một tác nhân thứ tư — **Người phản biện (Devil's Advocate)** — được chèn vào đồ thị ngay sau tác nhân Nhà phân tích định lượng, với nhiệm vụ được ràng buộc nghiêm ngặt và hẹp về phạm vi: **chỉ** tập trung lọc nhiễu và tìm điểm yếu trong chính nhận định xu hướng của Nhà phân tích định lượng — không được phép tự đề xuất một hướng phân tích mới, không được can thiệp vào các thông số quản trị rủi ro (vốn thuộc thẩm quyền của Nhà quản trị rủi ro) — nhằm giữ phạm vi trách nhiệm của tác nhân này tối giản và dễ kiểm chứng, tránh việc bổ sung một tác nhân "vạn năng" làm tăng độ bất định của toàn hệ thống.

Về mặt vận hành đồ thị, sau khi Người phản biện hoàn thành đánh giá, đồ thị rẽ nhánh có điều kiện: nếu Người phản biện xác định nhận định ban đầu có cơ sở vững (không phát hiện điểm yếu trọng yếu), luồng xử lý đi tiếp thẳng đến Nhà quản trị rủi ro như kiến trúc tuần tự hiện tại; nếu Người phản biện phát hiện điểm yếu đáng kể (ví dụ nhận định dựa trên một tín hiệu đơn lẻ chưa được đối chiếu chéo, hoặc bỏ qua một tín hiệu cảnh báo rủi ro rõ ràng từ dữ liệu đầu vào), đồ thị **quay lại** (loop back) tác nhân Nhà phân tích định lượng kèm theo phản biện cụ thể, yêu cầu tinh chỉnh nhận định trước khi tiếp tục — với một giới hạn số vòng lặp tối đa tường minh (ví dụ tối đa hai lượt xét lại) để tránh vòng lặp vô hạn và giữ độ trễ tổng thể trong một chặn trên có thể dự đoán được, cân bằng giữa lợi ích tự sửa lỗi và chi phí độ trễ đã phân tích ở mục 2.2.3.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ ĐỒ THỊ TRẠNG THÁI LANGGRAPH (QUANT ANALYST ⇄ DEVIL'S ADVOCATE → RISK MANAGER → EXECUTION JUDGE) TẠI ĐÂY]`

### 5.3.2. Pipeline tải lại nóng không gián đoạn (Zero-Downtime Hot-Reload)

Đề xuất khắc phục hạn chế đã nêu ở mục 5.2 và 3.1.2 — mô hình hiện chỉ được nạp một lần khi tiến trình khởi động và giữ cố định trong bộ nhớ — bằng một cơ chế **luồng thăm dò nền (background poller thread)** chạy song song với tiến trình phục vụ API chính, không chặn luồng xử lý yêu cầu:

1. Luồng thăm dò định kỳ (ví dụ mỗi 60 giây) đọc lại tệp kê khai `manifest/latest.json` (mục 3.1.2) và so sánh trường `active_model_s3_uri` với giá trị đang được instance mô hình hiện hành sử dụng.
2. Nếu phát hiện thay đổi, luồng thăm dò tải artifact mô hình mới **vào một vùng bộ nhớ riêng biệt**, hoàn toàn tách biệt khỏi instance mô hình đang phục vụ yêu cầu — quá trình tải diễn ra hoàn toàn trong nền, không ảnh hưởng đến các yêu cầu `/predict` đang được xử lý đồng thời bởi instance mô hình cũ.
3. Sau khi mô hình mới được nạp và xác minh thành công (ví dụ chạy thử một lượt suy luận trên dữ liệu mẫu để đảm bảo không có lỗi runtime), hệ thống thực hiện một **thao tác hoán đổi con trỏ nguyên tử** (atomic pointer swap) — cập nhật biến tham chiếu toàn cục mà các luồng xử lý yêu cầu đọc vào, từ instance mô hình cũ sang instance mô hình mới, trong một thao tác không thể bị ngắt quãng giữa chừng (đảm bảo bằng cơ chế khóa hoặc bằng tính nguyên tử sẵn có của việc gán tham chiếu đối tượng trong runtime Python).
4. Instance mô hình cũ chỉ được giải phóng khỏi bộ nhớ sau khi xác nhận không còn yêu cầu nào đang tham chiếu đến nó (ví dụ chờ hết một khoảng thời gian ân hạn ngắn sau thời điểm hoán đổi con trỏ).

Cơ chế này cho phép việc "quảng bá" một phiên bản mô hình mới qua tệp kê khai (đã có sẵn, mục 3.1.2) có hiệu lực **ngay lập tức trên tiến trình đang chạy**, không cần khởi động lại tiến trình dịch vụ — loại bỏ hoàn toàn khoảng thời gian gián đoạn hiện tại giữa lúc quảng bá mô hình mới và lúc tiến trình được khởi động lại theo chu kỳ triển khai kế tiếp.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ CƠ CHẾ HOT-RELOAD (BACKGROUND POLLER → TẢI MÔ HÌNH MỚI SONG SONG → HOÁN ĐỔI CON TRỎ NGUYÊN TỬ) TẠI ĐÂY]`

### 5.3.3. Bảo mật Zero-Trust nâng cao (Advanced DevSecOps)

Đề xuất bổ sung ba lớp phòng thủ đã được xác định là định hướng mở rộng ở mục 2.2.4, cụ thể hóa thành các hạng mục triển khai:

**Tường lửa ứng dụng web (AWS WAF) gắn vào CloudFront.** Đính kèm một Web ACL vào phân phối CloudFront hiện có (mục 2.3.3), áp dụng đồng thời các nhóm quy tắc quản lý sẵn có (AWS Managed Rules) cho các lớp tấn công phổ biến (SQL injection, XSS) và một quy tắc giới hạn tốc độ (rate-based rule) theo địa chỉ IP nguồn để giảm thiểu tấn công từ chối dịch vụ ở lớp ứng dụng — bổ sung một lớp lọc ở biên mạng, phía trước toàn bộ nhóm bảo mật EC2 hiện có, thay vì chỉ dựa vào việc EC2 không mở cổng SSH.

**Di chuyển token xác thực từ `localStorage` sang HttpOnly Cookie.** Như đã xác định ở mục 1.2.2 (chương giới thiệu) — nhắc lại tại đây với vai trò một đề xuất khắc phục cụ thể — token hiện được lưu tại `localStorage` phía trình duyệt (mục 2.3.3), có thể bị đọc bởi bất kỳ đoạn mã JavaScript nào chạy trên cùng trang (bao gồm mã độc chèn qua một lỗ hổng XSS nếu có). Đề xuất chuyển sang cơ chế cookie có cờ `HttpOnly` (không thể truy cập bằng JavaScript phía client), cờ `Secure` (chỉ gửi qua HTTPS), và thuộc tính `SameSite=Strict` hoặc `Lax` (giảm thiểu rủi ro giả mạo yêu cầu chéo trang — CSRF) — đòi hỏi điều chỉnh tương ứng ở tầng backend để đặt cookie tại thời điểm xác thực thành công và đọc token từ cookie thay vì tiêu đề `Authorization` tại tầng middleware xác thực Cognito (mục 3.2.2).

**S3 Object Lock (WORM) kết hợp xác minh mã băm mật mã học để chống đầu độc mô hình (Model Poisoning).** Bật chế độ Object Lock ở mức tuân thủ (compliance mode) cho các đối tượng trong kho lưu trữ mô hình học máy (mục 3.1.1), đảm bảo một artifact mô hình đã được công bố không thể bị ghi đè hoặc xóa trong suốt một khoảng thời gian lưu giữ tối thiểu — kể cả bởi thông tin đăng nhập quản trị có quyền cao nhất. Kết hợp với cơ chế xác minh: tại thời điểm công bố, artifact được ký bằng mã băm mật mã học (SHA-256) và giá trị băm được ghi vào một **danh sách cho phép đã được xác thực** (signed allow-list) tách biệt khỏi chính artifact; dịch vụ suy luận, trước khi cho phép `trust_remote_code=True` thực thi bất kỳ mã nào đi kèm artifact, tính lại mã băm của artifact vừa tải về và đối chiếu với danh sách cho phép — chỉ tiếp tục nạp mô hình nếu mã băm khớp. Cơ chế này nâng cấp trực tiếp từ cơ chế mã băm hiện tại (mục 2.4.1, 4.2) vốn chỉ phục vụ mục đích xác định cache lỗi thời, thành một **cổng kiểm soát bảo mật thực sự** trước khi cho phép thực thi mã nguồn từ một mô hình bên thứ ba.

`[CHÚ THÍCH: CHÈN SƠ ĐỒ MÔ HÌNH BẢO MẬT ZERO-TRUST NÂNG CAO (WAF → HTTPONLY COOKIE → S3 OBJECT LOCK + XÁC MINH MÃ BĂM) TẠI ĐÂY]`

### 5.3.4. Toán học tài chính: Tích hợp Value at Risk (VaR) cho quản trị rủi ro danh mục

Toàn bộ các thông số quản trị rủi ro hiện tại của Hội đồng cố vấn AI (mục 2.2.3, 3.2.3) — vùng vào lệnh, điểm dừng lỗ, chốt lời, đòn bẩy, kích thước vị thế — đều được tính toán **theo từng giao dịch đơn lẻ, độc lập với nhau**, không có một khung đánh giá rủi ro tổng hợp ở cấp độ danh mục đầu tư khi người dùng nắm giữ đồng thời nhiều vị thế trên nhiều tài sản. Đề xuất bổ sung một chỉ số **Giá trị chịu rủi ro (Value at Risk — VaR)** ở cấp độ danh mục, định lượng mức tổn thất tối đa kỳ vọng của toàn bộ danh mục trong một khung thời gian xác định (ví dụ 1 ngày) tại một mức độ tin cậy cho trước (ví dụ 95% hoặc 99%), theo phương pháp mô phỏng lịch sử (historical simulation) hoặc phương pháp phương sai-hiệp phương sai (variance-covariance) tận dụng trực tiếp dải biến động (volatility bands) đã có sẵn từ đầu ra của Chronos-2 (mục 2.2.1) cho từng tài sản trong danh mục, có tính đến hệ số tương quan giữa các tài sản (ví dụ mức độ tương quan giá giữa các đồng tiền mã hóa thường rất cao trong giai đoạn thị trường biến động mạnh, làm giảm hiệu quả đa dạng hóa danh mục theo cách trực giác thông thường). Chỉ số VaR danh mục này được đề xuất hiển thị như một thành phần bổ sung độc lập trên giao diện (mục 3.3), tách biệt khỏi khuyến nghị theo từng giao dịch của Hội đồng cố vấn AI, đóng vai trò một lớp cảnh báo rủi ro tổng thể ở tầm nhìn danh mục — hướng mở rộng có giá trị thực tiễn cao nhất cho người dùng nắm giữ đồng thời nhiều vị thế trên cả hai lớp tài sản đã xác lập ở mục 1.3.1.
