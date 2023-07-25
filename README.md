# DAY 12 – Data Engineering
## Оглавление
1. [Глава I](#глава-i) \
    1.1. [Преамбула](#преамбула)
2. [Глава II](#глава-ii) \
    2.1. [Общая инструкция](#общая-инструкция)
3. [Глава III](#глава-iii) \
    3.1. [Цели](#цели)
4. [Глава IV](#глава-iv) \
    4.1. [Задание](#задание)
5. [Глава V](#глава-v) \
    5.1. [Сдача работы и проверка](#сдача-работы-и-проверка)

## Глава I
### Преамбула
Очень важно писать код в продакшн в машинном обучении, потому что это позволяет создавать стабильные и надежные модели, 
которые могут быть использованы в реальных условиях. Код должен быть написан таким образом, чтобы он мог быть 
масштабирован и поддерживаться в долгосрочной перспективе. Это включает в себя использование хороших практик 
программирования, таких как комментирование кода, разделение кода на [модули и тестирование кода перед его развертыванием](https://drivendata.github.io/cookiecutter-data-science/).

Кроме того, важно следить за тем, чтобы код был безопасным и не содержал уязвимостей, которые могут быть использованы 
злоумышленниками для атаки на систему. В целом, написание хорошего кода в продакшн в машинном обучении является ключевым 
фактором для создания успешных и эффективных моделей, которые могут быть использованы в реальной жизни.

Несмотря на то, что Jupyter Notebook является популярным инструментом для разработки и прототипирования кода в 
машинном обучении, его использование в продакшн не рекомендуется по [нескольким причинам](https://towardsdatascience.com/5-reasons-why-jupyter-notebooks-suck-4dc201e27086):
* Jupyter Notebook не является надежным и безопасным инструментом для работы в продакшн. Он не предназначен для запуска 
кода в производственной среде и может привести к ошибкам и уязвимостям в безопасности.
* Jupyter Notebook не поддерживает масштабирование и не позволяет легко переносить код в другие среды. 
Он не обеспечивает возможность управления зависимостями и версиями, что может привести к проблемам при развертывании 
кода в других средах.
* Jupyter Notebook не предоставляет возможность для автоматического тестирования кода, что может привести к 
ошибкам и проблемам в производственной среде.

В целом, хотя Jupyter Notebook может быть полезным инструментом для разработки и прототипирования кода в машинном 
обучении, его использование в продакшн не рекомендуется из-за его ограниченных возможностей по безопасности, 
масштабированию и тестированию кода.

![jupyter](./misc/images/jupyter.png)

## Глава II
### Общая инструкция

Методология Школы 21 может быть не похожа на тот образовательный опыт, который случался с тобой ранее. Её отличает высокий уровень автономии: у тебя есть задача, ты должен её выполнить. По большей части тебе нужно будет самому добывать знания для её решения. Второй важный момент — это peer-to-peer обучение. В образовательном процессе нет менторов и экспертов, перед которыми ты защищаешь свой результат. Ты это делаешь перед таким же учащимися, как и ты сам. У них есть чек-лист, который поможет им качественно выполнить приемку вашей работы.

Роль Школы 21 заключается в том, чтобы обеспечить через последовательность заданий и оптимальный уровень поддержки такую траекторию обучения, при которой ты не только освоишь hard skills, но и научишься самообучаться.

- Не доверяй слухам и предположениям о том, как должно быть оформлено ваше решение. Этот документ является единственным источником, к которому стоит обращаться по большинству вопросов;
- твое решение будет оцениваться другими учащимися;
- подлежат оцениванию только те файлы, которые ты выложил в GIT (ветка develop, папка src);
- в твоей папке не должно быть лишних файлов — только те, что были указаны в задании;
- не забывай, что у вас есть доступ к интернету и поисковым системам;
- обсуждение заданий можно вести и в Rocket.Chat;
- будь внимателен к примерам, указанным в этом документе — они могут иметь важные детали, которые не были оговорены другим способом;
- и да пребудет с тобой Сила!



## Глава III
### Цели
Твоего коллегу, ответственного за введение алгоритмов "в production" [сбил автобус](https://dblinov.com/blog/tpost/x4y637mh31-bus-factor-faktor-avtobusa).
И теперь тебе придется завершить его работу. Он успел структурировать проект, дописал несколько функций, но многое не успел.
Тебе потребуется разобраться в проделанной им работе и реализовать "production ready" проект для решения задачи классификации болезни сердца. 
Требуется:
* дописать модуль для обучения модели 
* покрыть этот модуль тестами
* с помощью этого модуля обучить разные модели классификации
* сохранить артефакты обучения (логи обучения, обученную модель, метрики качества)

## Глава IV
При выполнении этого проекта ты забываешь о jupyter-notebooks. Мы будем писать структурированный модуль обучения модели. 
Информацию о модуле, структуре проекта и настройке окружения твой коллега оставил [тут](src/README.md). Скачай датасет и настрой
виртуальное окружение согласно [инструкции](src/README.md). Некоторая часть кода в этом проекте уже написана, некоторую 
предстоит написать самому, а некоторая даже содержит ошибки. Придется разбираться.

### Task 1
Для возможности гибкой настройки процесса обучения в этом проекте коллега использовал библиотеку [hydra](https://hydra.cc/docs/intro/).
Изучи эту библиотеку, разберись как она работает и как связана с библиотекой [dataclasses](https://docs.python.org/3/library/dataclasses.html).
Поправь конфиги в папке `configs`.

### Task 2
Для обучения планировалось обучить логистическую регрессию и алгоритм случайного леса. Для логистической регрессии 
`dataclass` параметров модели [написан](src/heart/entities/models.py), а для случайного леса нет. 
Напиши `dataclass` для параметров модели случайного леса. Расставь [type hints](https://docs.python.org/3/library/typing.html) для класса.

### Task 3
Допиши функцию [split_train_test_data](./src/heart/data/make_dataset.py). Разберись, что она должна принимать на вход, а 
что должна возвращать. Расставь type hints. Используй эту функцию в общем [pipeline обучения](./src/heart/train.py).
В логах обучения выведи размеры тренировачной и тестовой выборки.

### Task 4
Хорошо бы проверить насколько правильно работает наша функция разделения. 
Допиши тест на функцию [test_split_dataset](./src/tests/data/test_make_dataset.py).

### Task 5
Теперь перейдем к реализации преобразования данных. Разбериcь как работает [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
Добавь в pipeline категориальных признаков заполнение пропусков с помощью функции [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
Убедись, что [test_process_categorical_features](./src/tests/features/test_make_categorical_features.py) проходит проверку.

### Task 6
Теперь перейдем к численным признакам. По аналогии с pipeline для категориальных признаков добавь в функцию [build_transformer](./src/heart/features/build_features.py) 
pipeline для численных признаков. В функцию [build_numerical_pipeline](./src/heart/features/build_features.py) добавь преобразование 
[class OutlierRemover](./src/heart/features/build_features.py) и еще одно преобразование на твое усмотрение. 

### Task 7
Убедись, что pipeline обучения моделей проходит все тесты `pytest tests`.

### Task 8
Теперь проверим стиль нашего кода с помощью [линтера](https://habr.com/ru/companies/oleg-bunin/articles/433480/).
Воспользуемся литером [Pylint](https://pylint.pycqa.org/en/latest/) - `pylint heart`. Набери качество кода больше 8/10.

### Task 9
Теперь мы добавили все, что требуется для обучения моделей, их логирования и сохранения. Запустим наше обучение.
Логистической регрессии по инструкции в [README](./src/README.md). Посмотри папку с артефактами. Там должны сохраниться логи обучения, сохраненная 
модель, метрики качества. Pipeline обучения должен работать корректно. 
Сохрани артефакты последнего запуска обучения для модели Логистической регрессии.

### Task 10
Дополни инструкцию в [README](./src/README.md) недостающей информацией (Она помечена `?`). Напиши инструкцию для обучения 
модели случайного леса. Обучи эту модель. Посмотри папку с артефактами. Там должны сохраниться логи обучения, сохраненная 
модель, метрики качества. Pipeline обучения должен работать корректно. 
Сохрани артефакты последнего запуска обучения для модели Случайного леса.


## Глава V
### Сдача работы и проверка
Сохрани решенный проект в src.\
Загрузи изменения на Git в ветку develop.\
Во время проверки проверяемый должен показать, что код проходит тесты `pytest heart` и показать качество кода через 
линтер `pylint heart`.

💡 [Нажми здесь](https://forms.gle/z1TzYKxiwd8C4cxm9) **чтобы отправить обратную связь по проекту**. 