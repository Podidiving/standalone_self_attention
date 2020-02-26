# standalone self attention
[arxiv](https://arxiv.org/pdf/1906.05909.pdf)

## Operation counting, using [Thop](https://github.com/Lyken17/pytorch-OpCounter)
![Operation_counting](./pics/op_count.png)

## Time consuming tests
![Time_consuming](./pics/time_consuming.png )

# How to train
## Classification
Just configure `./configs/classification_config.yml` (it has lot's of comments)
And run `./scripts/classification_train.sh`
    
If you don't have csv files, you can run `./datautils/make_csv.py` with required options
You must pass `root` folder to this script. Structure must be next:
```
root/
    category_1/
        picture_1.jpg
        ...
        picture_n.jpg
    ...
    category_m/
        picture_1.jpg
        ...
        picture_n.jpg

```