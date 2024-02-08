import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = [
    "quality product and very fast delivery", #https://www.amazon.com/-/es/gp/customer-reviews/R3CWNSY2AEOX1A/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR#
    "Wore this for a few weeks. Was doing good until the zipper broke! Don't waste your money!", #https://www.amazon.com/gp/customer-reviews/R1QWR6PY1JD6B8/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "The best you can get now", #https://www.amazon.com/-/es/gp/customer-reviews/R3GRKOZ33MNF2J/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "the zipper on this jacket is awful. i want my money back.", #https://www.amazon.com/gp/customer-reviews/R3FKUSOL40JJQ9/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "The most amazing Garmin ever had. Highly recommend guys!", #https://www.amazon.com/gp/customer-reviews/R29UH83K2OMIIL/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "My rugs are softer than this fleece. Do not order!", #https://www.amazon.com/gp/customer-reviews/RCYJQPI6RFJIT/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Fast delivery and great quality product.", #https://www.amazon.com/gp/customer-reviews/RWJ9JQDWVCG7P/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "Merchandise is too small. Large is really a medium.", #https://www.amazon.com/gp/customer-reviews/R3TW4SGMHR9GWN/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Nice product!", #https://www.amazon.com/-/es/gp/customer-reviews/R2U71AZ84QC8TU/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "Didn't like this at all. Have returned it.", #https://www.amazon.com/gp/customer-reviews/R3IIZ1WFYLZLSF/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Excellent price. Very comfortable. Long lasting battery. Excellent quality. Easy to read.", #https://www.amazon.com/-/es/gp/customer-reviews/R3IU7QP9Q380VO/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "Warm jacket but after 6 weeks of light use, the zipper is completely unusable. Would not recommend.", #https://www.amazon.com/gp/customer-reviews/R3VYODOFGV12J3/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "I am using it every day for workout and for a cycling Maps.", #https://www.amazon.com/product-reviews/B0BYFCW2GR/ref=cm_cr_getr_d_paging_btm_next_4?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews&pageNumber=4#reviews-filter-bar
    "Husband didn’t like it", #https://www.amazon.com/gp/customer-reviews/R27GZE9734FB3H/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Super great phone. I could say that this seller is a reliable seller. Brand new unused Samsung S23 Ultra 512 factory unlocked at a good pricing. Supet fast shipment.", #https://www.amazon.com/gp/customer-reviews/R1LEO66UUKWVSD/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "The zipper doesn't work on this. This makes it completely unusable.", #https://www.amazon.com/gp/customer-reviews/R1OL8TFIR6GXUC/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Works super fast. With the large screen, I can easily read and watch stuff that I like.", #https://www.amazon.com/gp/customer-reviews/R36GSYGHDPFUQE/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Not what i expected", #https://www.amazon.com/gp/customer-reviews/R2M6ESW5P9IC5A/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Insane camera specially when you whip out the Pro Mode", #https://www.amazon.com/gp/customer-reviews/RH17B3V8H4M0P/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "cheap material, not very well made.", #https://www.amazon.com/gp/customer-reviews/R1S15WRPWGBBZ8/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Love this phone the price was good the battery last me all day", #https://www.amazon.com/gp/customer-reviews/R9Y8I0QH0NQAI/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N 
    "Having to return this. Seal was broken. I have to send it back and get another one.", #https://www.amazon.com/gp/customer-reviews/R10NU261LFEJQW/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "awesome camera and fast phone with good battery life", #https://www.amazon.com/gp/customer-reviews/R1NMWOIDXD8QG4/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "The product arrived all open and in poor condition.", #https://www.amazon.com/gp/customer-reviews/R3GMZF4BT23UOZ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "Expectations definitely met", #https://www.amazon.com/gp/customer-reviews/R3VA770BMJ4X8Y/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "The quality of Samsung's S23 Ultra screen is low. The phone gets scratch marks out of nowhere.", #https://www.amazon.com/gp/customer-reviews/R348PEWG2CDY3G/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Nice and so far so good. My husband loves it", #https://www.amazon.com/gp/customer-reviews/R3HA51N2PTS7P8/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Ordered on Cyber Monday sale, it came today and it was 'renewed.' Not what I ordered", #https://www.amazon.com/gp/customer-reviews/R3VIN0Z2AJRVG7/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Great buy", #https://www.amazon.com/gp/customer-reviews/RYTACHB3H1P2E/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Received a defective phone.", #https://www.amazon.com/gp/customer-reviews/RIUDIO3Z3ZEL6/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "We received the item super fast and it was new. We love it!!! I highly recommend purchasing from this person/company. Thank you!!!", #https://www.amazon.com/gp/customer-reviews/RTPN4MWYPRDV/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6 
    "There was nothing to like about it wasn't working", #https://www.amazon.com/gp/customer-reviews/R69YJMU0Z9KXJ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "So fast, so quiet and way better graphics and controllers.", #https://www.amazon.com/gp/customer-reviews/R330ZUCB6U0SBV/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "Ordered but never received it.", #https://www.amazon.com/gp/customer-reviews/RN6HEIYOCAOWV/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0BLP2PY6N
    "Fast delivery came next day everything is amazing", #https://www.amazon.com/gp/customer-reviews/R33TVNY8T5SZ2O/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "Dont fit in my ears, fall out immediately. Very disappointed. Also hard to get out of the case.", #https://www.amazon.com/gp/customer-reviews/R3ESDR0TGVKNCA/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CCK1T2GN
    "I can only say that it is the best console.", #https://www.amazon.com/gp/customer-reviews/R2VZBID245TWBQ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "The 1 star is for the shipment issue, ended up returning.", #https://www.amazon.com/gp/customer-reviews/RFZ5MIOTNZUL/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CCK1T2GN
    "My husband loves it.", #https://www.amazon.com/gp/customer-reviews/R1J0YTZJHPLCHO/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "They weren't that great in the first place but what a waste of $200.", #https://www.amazon.com/gp/customer-reviews/R1DP0XRHBE0K0Y/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CCK1T2GN
    "Everything was there and it works fine and is smaller tham my older ps5 so im happy", #https://www.amazon.com/gp/customer-reviews/R2NZE8PP34XS1U/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "Very dark screen, made a return, amazon has not refunded the money yet.", #https://www.amazon.com/-/es/gp/customer-reviews/R3HZVPALNTQHHO/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0BYFCW2GR
    "Worked perfectly my boyfriend has been playing his new game all day perfect gift", #https://www.amazon.com/gp/customer-reviews/R2CYDMYZSQ0O51/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CKZGY5B6
    "Picture speaks for itself… very disappointing :(", #https://www.amazon.com/-/es/product-reviews/B0BYFCW2GR/ref=acr_dp_hist_1?ie=UTF8&filterByStar=one_star&reviewerType=all_reviews#reviews-filter-bar
    "this jacket is great for jogging, etc. It's made well.", #https://www.amazon.com/gp/customer-reviews/R18ZE7QLGGL0IF/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "The worst nba 2k ever made! Do not waste your money.", #https://www.amazon.com/-/es/gp/customer-reviews/R2EJVKTBY4LE1C/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0CBL4DMNZ 
    "My Son loved this Jacket. It is light yet feels very warm & practical. Great buy & money well spend.", #https://www.amazon.com/gp/customer-reviews/R1883BSXRVI86M/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Battery dies in 30 seconds, the razor is so dull and leaves cuts in your skin and this device is not worth it.", #https://www.amazon.com/gp/customer-reviews/R3I4O7B10283IA/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B08ZYYQ8S9
    "Fits great and is very comfortable. Nice for cool evenings.", #https://www.amazon.com/gp/customer-reviews/R3CLX62YB1U0BG/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0738J92CM
    "Cheap material, the size is tight.",#https://www.amazon.com/gp/customer-reviews/R2NTI2VUN9EW8Y/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B0CM13KV1C
    "This is exactly exactly what I was looking for", #https://www.amazon.com.mx/product-reviews/B0BKW74N7G/ref=cm_cr_unknown?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar
    "Materials are not good and stich is not good", #https://www.amazon.com.mx/product-reviews/B0BKW74N7G/ref=cm_cr_unknown?ie=UTF8&filterByStar=one_star&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar
    "Nice looking slipcover to extend the life of my comfortable old Ethan Allen sofa", #https://www.amazon.com/product-reviews/B0CJTX53TS/ref=cm_cr_unknown?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews&pageNumber=1#reviews-filter-bar
    "Did not fit on my couch", #https://www.amazon.com/gp/customer-reviews/R3GPZW8C8GPTMM/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0CJTX53TS
    "He was so excited it came with exchangeable color key caps.", #https://www.amazon.com/-/es/gp/customer-reviews/R38UGZNC9NHSFC/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Would not recommend, tears easily cheaply made", #https://www.amazon.com/-/es/gp/customer-reviews/RXPXGBSRRR99M/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0CJTX53TS
    "Was a Christmas gift and he loved it. It’s a tiny bit loud. Very light weight and easy to use really nice quality", #https://www.amazon.com/-/es/gp/customer-reviews/RFJM91ATKG5B/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Worst couch cover in the market! After one wash, fabric fell apart like snow flakes.", #https://www.amazon.com/-/es/gp/customer-reviews/R2IY487HOXT0NH/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B0CJTX53TS
    "Good price and quality", #https://www.amazon.com/gp/customer-reviews/R187ZPTY90V0L2/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Super disappointed, worked well on the first try but now it’s junk and it does not function", #https://www.amazon.com/-/es/gp/customer-reviews/R24OTAICUCRF5L/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "for the price, this is a great keyboard.", #https://www.amazon.com/gp/customer-reviews/R2WFJILYH1ZB6X/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Almost a year old and lights have stopped working don’t think I’d buy again", #https://www.amazon.com/-/es/gp/customer-reviews/R2REIQIBUQFYGY/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "It’s good I do recommend", #https://www.amazon.com/gp/customer-reviews/RXQ6G61PWTWQX/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "I have received the package; unfortunately, it has been damaged. ", #https://www.amazon.com/-/es/gp/customer-reviews/R3Q3HSKNJL3NNW/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "This keyboard is amazing, this is my first keyboard bought from amazon. Excellent packaging and delivery!!", #https://www.amazon.com/gp/customer-reviews/RU9D30HJ98O39/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Every time I try to use my “W” key to move, it doesn’t do the job right for me.", #https://www.amazon.com/gp/customer-reviews/R14CACGAW8128R/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Loving this keyboard definitely worth every penny super affordable and works amazing ", #https://www.amazon.com/gp/customer-reviews/R12DFCJTN55ZSB/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "too loud", #https://www.amazon.com/gp/customer-reviews/R2EX3TQN8D7MIQ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "It met our expectations", #https://www.amazon.com/gp/customer-reviews/R2649F72ZGDYWU/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "The A key was not functioning", #https://www.amazon.com/gp/customer-reviews/RWKIL05X08D9B/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "The keyboard has a low travel distance and is fantastic for gaming.", #https://www.amazon.com/gp/customer-reviews/R1DUJF37FXE1M3/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "the keyboard barely lasted me 1 month and 2 keys no longer work and cannot be replaced. don't buy this", #https://www.amazon.com/gp/customer-reviews/R2KD7WF0Y5M8DJ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "sexy keyboard", #https://www.amazon.com/gp/customer-reviews/R1UPJ8XAHM39Y6/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "I had to throw out the keyboard after a month, I do not recommend at all.", #https://www.amazon.com/gp/customer-reviews/R24AQ7TCKKI54Q/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "funny keyboard", #https://www.amazon.com/gp/customer-reviews/R8O9BHND2IJGY/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Cheap, poor quality.", #https://www.amazon.com/gp/customer-reviews/R23L0BFGJRG0EE/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Can’t beat the price, great sound great feel. Good features considering the price point.", #https://www.amazon.com/gp/customer-reviews/R1EYSTRFONW1L6/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Keyboard arrived with many broken switches.", #https://www.amazon.com/gp/customer-reviews/R2U5LX99Q5Z5R2/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "the feeling of the keys is so nice the lgb light are cool too.", #https://www.amazon.com/gp/customer-reviews/R2GZ3HH75N9JNW/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "In the first day when I got my keyboard it was broken right away I would have to plug it in 3 times so it can work normally", #https://www.amazon.com/gp/customer-reviews/R2ZFTAD712OGJD/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "My son loved it, great feel when typing, great quality and value, love the lighting", #https://www.amazon.com/gp/customer-reviews/R2JVCMKMUGUIWK/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "The keys stop functioning properly after a couple of weeks of testing it.", #https://www.amazon.com/gp/customer-reviews/R5MPCOBHX7FR9/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Very easy use. Great quality. Would buy again.", #https://www.amazon.com/gp/customer-reviews/R1FZ84TL29NFVY/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "The keys just stopped working in two months, terrible product.", #https://www.amazon.com/gp/customer-reviews/R20HWJSTMBZZML/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "This is a nice cover at a reasonable price. It looks good and even feels good too.", #https://www.amazon.com/-/es/gp/customer-reviews/R3G1CN1J4JBFKX/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "Wish I did not buy this.", #https://www.amazon.com/gp/customer-reviews/R1J8JJCHSJ8169/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "I wanted a simple, affordable, functioning cover for my iPad. This is it.", #https://www.amazon.com/gp/customer-reviews/RB8NAMHVC61YJ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "Keys stopped lighting up after a few days. Super cheap. Stay away.", #https://www.amazon.com/gp/customer-reviews/RWOV4LT4KZKK6/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "I wanted a simple, affordable, functioning cover for my iPad. This is it.", #https://www.amazon.com/gp/customer-reviews/RB8NAMHVC61YJ/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "The keyboard broke within a week of having it.", #https://www.amazon.com/gp/customer-reviews/R20BFGOZ3GXGOX/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B098LG3N6R
    "Easy to put on. Nice material. Inexpensive! just make the easy choice and buy this!", #https://www.amazon.com/gp/customer-reviews/R1Y5UOZMKHXS2/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "Fell apart on day one", #https://www.amazon.com/-/es/gp/customer-reviews/R34FRAP03P21E1/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B08KGLTN9Q
    "I got this for my son and he is happy with it.", #https://www.amazon.com/gp/customer-reviews/R1OPXDWGTICUK8/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "This is a piece of garbage. Buyer beware. It doesn’t even turn on. And it will take another 30 days to get a refund. Do not waste you’re money!", #https://www.amazon.com/-/es/gp/customer-reviews/RBARD2ZUUBYD3/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B08KGLTN9Q
    "I just got a new iPad and this is the perfect case!!", #https://www.amazon.com/gp/customer-reviews/R28Z9XQ8G5N7ID/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "I would’ve like it if I didn’t arrive completely shattered", #https://www.amazon.com/gp/customer-reviews/R2E4P4PFHQ3P96/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B08GK417D4
    "I like it, it protected and lets you stand it up", #https://www.amazon.com/gp/customer-reviews/R1EY2TDRJY1AE6/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "Button up arrived with dirt and some black stains on it", #https://www.amazon.com/-/es/gp/customer-reviews/R3BOFUDD44XOTJ/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B09LKNT3H7
    "I liked this because it was easy to insert my iPad", #https://www.amazon.com/gp/customer-reviews/RRK4EQLWFSPUE/ref=cm_cr_getr_d_rvw_ttl?ie=UTF8&ASIN=B07XY28FZG
    "The product is not at all as it is described and shown on your sight not worth the money spend." #https://www.amazon.com/-/es/gp/customer-reviews/R27IO7L5SGZQFE/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B09LKNT3H7


]

labels = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]


tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences).tolist()

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(word_index) + 1, output_dim=16),
    keras.layers.SimpleRNN(8),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)


new_sentences = ["quality product and very fast deliver"]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, maxlen=len(padded_sequences[0]))


prediction = model.predict(new_padded_sequences)

if prediction[0][0] >= 0.5:
    print("Es un comentario positivo")
else:
    print("Es un comentario negativo")
