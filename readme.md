
<!-- ABOUT THE PROJECT -->
## Style Tranfer Project

Style Transfer là một bài toán biến đổi hình hành theo các style mong muốn. Trong bài toán này, mình sử dụng model VGG19 để lấy thông tin features tại từng convolution layer.

###  Style Tranfer Technical
+ Sử dụng pretain model VGG19.
+ Sử dụng VGG19 để lấy thông tin features tại các convolution layer.
+ Xây dựng hàm loss:  wieght_1 x content loss (bức ảnh gốc so với ảnh mục tiêu) + wieght_2 x style_loss (bức ảnh style so với ảnh mục tiêu)
+ Update gradient ảnh target


### Built  Environment

Setup môi trường và chạy server

<!-- GETTING STARTED -->
## Getting Started



### Prerequisites
    *python
    ```
    python 3.8.5
    ```
### Installation

1. Clone the repo
   ```
   git clone https://github.com/Phammanh26/style-transfer.git
   ```
3. tạo môi trường `conda` hoặc `virtualenv`, ví dụ tên môi trường: `venv`
   *conda
   ```
   conda create -n venv python=3.8.5
   ```
   *virtualenv
   ```
   virtualenv venv
   ```

3. Activate môi trường:
   *conda
   ```
   conda activate venv
   ```
   *virtualenv
   ```
   source venv\bin]activate

4. Tải thư viện:
    ```
    pip install -r requirements/dev.txt
    ```
5. chạy server:

    *step1:
    ```
    cd src
    ```

    *step2:
    ```
    sh start_script.sh
    ```


<!-- USAGE EXAMPLES -->
## Usage

Sau quá trình cài thư viên và chạy server hoàn tất, bạn có thể vào đường link: ```http://0.0.0.0:8000/docs``` để thử nghiệm.

<!-- CONTACT -->
## Contact

Manh Pham - pvm26042000@gmail.com

Project Link: [https://github.com/Phammanh26/style-transfer.git](https://github.com/Phammanh26/style-transfer.git)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Một số kiến thức liên quan để có thể thực hiện 
* [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf?fbclid=IwAR1W6LEGRg7AhH-EpKJ8SHGqIZ06N6TtaQUALph7g509V84zTB7Rx46l5eY)
