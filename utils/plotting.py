import multiprocessing as mp
import logging
import cv2
from .timing import timeit


def bbplot_img_from_df(df):
    imagefilename = df['file_name'].iloc[0]
    savefilename = df['save_name'].iloc[0]
    try:
        img = cv2.imread(imagefilename)
    except:
        logging.FATAL("The image {} could not be read.".format(imagefilename))

    plotonimage(img, df, savefilename)
    return None


@timeit
def plot_detections_on_db(bboxframe):
    assert 'save_name' in bboxframe.columns.values, "There is no column " \
                                                    "called save_name in the " \
                                                    "DataFrame."

    unique_filenames = bboxframe['file_name'].unique().tolist()

    pool = mp.Pool(processes=128)

    results = [pool.apply(bbplot_img_from_df,
                          args=(bboxframe.loc[bboxframe['file_name'] == x],))
               for x in unique_filenames]

    return None

def plotonimage(img, df, savefilename):
    for bbox, category in zip(df['bbox'], df['category_name']):
        cv2.rectangle(img, pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2]
                                                        - 1, bbox[1] + bbox[
                                                            3] - 1),
                      color=(0, 255, 255), thickness=2)

        box, _ = cv2.getTextSize(category, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 3)
        cv2.rectangle(img, (bbox[0], bbox[1]),
                      (bbox[0] + box[0], bbox[1] + box[1] + 5),
                      color=(0, 255, 255), thickness=-1)
        cv2.putText(img, category, (bbox[0], bbox[1] + box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    cv2.imwrite(savefilename, img)

    return None


